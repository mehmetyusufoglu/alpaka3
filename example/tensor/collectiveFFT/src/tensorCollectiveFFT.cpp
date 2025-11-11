/**
 * Tensor Collective FFT
 * ----------------------
 * Distributed FFT demo built on Alpaka's tensor collective provider.
 * The executable partitions a very large 1-D signal across MPI/NCCL ranks
 * using a strided ownership scheme, computes the per-rank spectrum, applies
 * the phase shift implied by each rank's lag, and finally uses NCCL to sum
 * the phase-corrected spectra so every participant reconstructs the global FFT.
 *
 * The focus is on demonstrating how to compose "FFT of parts" on multiple GPUs
 * rather than on peak FFT performance; each rank runs a naive O(n^2) DFT, applies
 * the phase correction implied by its stride offset, and NCCL sums the complex
 * contributions into the global spectrum. Production builds can replace the local
 * DFT with cuFFT (or another vendor FFT) and slot into a multi-node cuFFT strategy
 * once the orchestration pattern shown here is validated.
 */
#include "collectiveFft/bootstrap.hpp"
#include "collectiveFft/dataSource.hpp"
#include "collectiveFft/options.hpp"
#include "collectiveFft/verification.hpp"

#include <alpaka/alpaka.hpp>
#include <alpaka/onHost/example/executors.hpp>
#include <alpaka/onHost/executeForEach.hpp>
#include <alpaka/onHost/interface.hpp>
#include <alpaka/tensor/core/TensorTypes.hpp>
#include <alpaka/tensor/providers/CleanTensorOpContext.hpp>
#include <alpaka/tensor/providers/EnabledVendorLibs.hpp>
#include <alpaka/tensor/providers/collective/CollectiveTypes.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace tt = alpaka::tensor;
namespace collective = alpaka::tensor::collective;

namespace
{
    using collectiveFft::CommandLineOptions;
    using collectiveFft::compareSpectra;
    using collectiveFft::computeLocalFft;
    using collectiveFft::computeLocalFftWithPrecision;
    using collectiveFft::DirectVerifyPrecision;
    using collectiveFft::loadFullSignalSequence;
    using collectiveFft::loadReferenceSpectrumFromFile;
    using collectiveFft::loadSignalChunk;
    using collectiveFft::MultiProcessBootstrap;
    using collectiveFft::VerificationStats;

    [[nodiscard]] std::optional<int> parseEnvInt(char const* envName)
    {
        if(char const* value = std::getenv(envName))
        {
            char* end = nullptr;
            long parsed = std::strtol(value, &end, 10);
            if(end != nullptr && end != value && *end == '\0' && parsed >= 0)
            {
                return static_cast<int>(parsed);
            }
        }
        return std::nullopt;
    }

    [[nodiscard]] std::optional<int> discoverLocalRank(MultiProcessBootstrap const& bootstrap)
    {
        if(bootstrap.localRank >= 0)
            return bootstrap.localRank;

        static constexpr std::array<char const*, 4> envKeys{
            "OMPI_COMM_WORLD_LOCAL_RANK",
            "MPI_LOCALRANKID",
            "SLURM_LOCALID",
            "LOCAL_RANK"};
        for(auto const* key : envKeys)
        {
            if(auto const value = parseEnvInt(key))
                return value;
        }
        return std::nullopt;
    }

    // Version 1 pipeline step 4: compose the global spectrum contribution for this rank using phase correction.
    std::vector<std::complex<float>> buildRankContribution(
        std::span<std::complex<float> const> localSpectrum,
        int worldRank,
        std::size_t worldSize,
        std::size_t totalSamples)
    {
        std::size_t const samplesPerRank = localSpectrum.size();
        std::vector<std::complex<float>> contributions(totalSamples, std::complex<float>{0.0f, 0.0f});
        if(samplesPerRank == 0 || totalSamples == 0)
            return contributions;

        constexpr float twoPi = 2.0f * std::numbers::pi_v<float>;
        for(std::size_t q = 0; q < samplesPerRank; ++q)
        {
            float angleBase
                = -twoPi * static_cast<float>(worldRank) * static_cast<float>(q) / static_cast<float>(totalSamples);
            std::complex<float> const basePhase = std::polar(1.0f, angleBase);

            for(std::size_t t = 0; t < worldSize; ++t)
            {
                float angleAcross
                    = -twoPi * static_cast<float>(worldRank) * static_cast<float>(t) / static_cast<float>(worldSize);
                std::complex<float> const acrossPhase = std::polar(1.0f, angleAcross);
                std::size_t const globalIndex = q + t * samplesPerRank;
                if(globalIndex >= contributions.size())
                    continue;

                contributions[globalIndex] = localSpectrum[q] * basePhase * acrossPhase;
            }
        }

        return contributions;
    }

    void printSpectrumPreview(int worldRank, std::vector<std::complex<float>> const& spectrum, std::size_t previewBins)
    {
        if(worldRank != 0 || spectrum.empty())
            return;

        std::size_t const bins = std::min(previewBins, spectrum.size());
        auto const originalFlags = std::cout.flags();
        auto const originalPrecision = std::cout.precision();
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "First " << bins << " frequency bins (real, imag, magnitude):\n";
        for(std::size_t k = 0; k < bins; ++k)
        {
            float magnitude = std::abs(spectrum[k]);
            std::cout << "  k=" << std::setw(4) << k << " : " << std::setw(12) << spectrum[k].real() << "  "
                      << std::setw(12) << spectrum[k].imag() << "i  |X|=" << magnitude << '\n';
        }
        auto maxIt = std::max_element(
            spectrum.begin(),
            spectrum.end(),
            [](std::complex<float> const& lhs, std::complex<float> const& rhs)
            { return std::abs(lhs) < std::abs(rhs); });
        if(maxIt != spectrum.end())
        {
            std::size_t const idx = static_cast<std::size_t>(std::distance(spectrum.begin(), maxIt));
            std::cout << "Dominant bin: k=" << idx << " |X|=" << std::abs(*maxIt) << '\n';
        }
        std::cout.flags(originalFlags);
        std::cout.precision(originalPrecision);
    }

    template<typename Tag>
    int runCollectiveFft(Tag const& tag, MultiProcessBootstrap const& bootstrap, CommandLineOptions const& options)
    {
        auto const& deviceSpec = tag[alpaka::object::deviceSpec];
        auto const& exec = tag[alpaka::object::exec];
        auto selector = alpaka::onHost::makeDeviceSelector(deviceSpec);
        if(!selector.isAvailable())
        {
            if(!bootstrap.enabled || bootstrap.worldRank == 0)
            {
                std::cout << "Skipping backend " << deviceSpec.getApi().getName() << " (no device available)\n";
            }
            return 0;
        }

        int deviceCount = selector.getDeviceCount();
        if(deviceCount <= 0)
        {
            deviceCount = 1;
        }

        int deviceId = 0;
        if(deviceCount > 0)
        {
            auto const preferredRank = discoverLocalRank(bootstrap).value_or(bootstrap.worldRank);
            deviceId = preferredRank % deviceCount;
        }

        auto device = selector.makeDevice(deviceId);
        auto queue = device.makeQueue();

        using Device = decltype(device);
        using Exec = std::decay_t<decltype(exec)>;

        if(!bootstrap.enabled || bootstrap.worldRank == 0)
        {
            std::cout << "\n=== tensor Collective FFT ===\n";
            std::cout << "API: " << deviceSpec.getApi().getName() << '\n';
            std::cout << "Executor: " << alpaka::onHost::demangledName(exec) << '\n';
            if(bootstrap.enabled)
            {
                std::cout << "World size: " << bootstrap.worldSize << '\n';
                if(bootstrap.localRank >= 0)
                {
                    std::cout << "Local size: " << bootstrap.localSize << '\n';
                }
            }
            else if(bootstrap.worldSize > 1)
            {
                std::cout << "MPI detected " << bootstrap.worldSize << " ranks but collectives stayed single-rank";
                if(!bootstrap.disableReason.empty())
                {
                    std::cout << " (" << bootstrap.disableReason << ")";
                }
                std::cout << '\n';
            }
        }

        if(!tt::EnabledVendorLibs::hasNCCL)
        {
            if(!bootstrap.enabled || bootstrap.worldRank == 0)
            {
                std::cout << "NCCL support not enabled in this build; skipping distributed FFT.\n";
            }
            return 0;
        }

        if constexpr(std::is_same_v<Exec, alpaka::exec::GpuCuda>)
        {
            auto context = tt::createCleanTensorOpContext(exec, device, queue);

            if(bootstrap.enabled)
            {
                std::cout << "Rank " << bootstrap.worldRank << " using device " << deviceId << '\n';
            }

            collective::GroupConfig groupConfig{};
            groupConfig.deviceIds.push_back(deviceId);
            if(bootstrap.enabled)
            {
                groupConfig.multiProcess = true;
                groupConfig.worldRank = bootstrap.worldRank;
                groupConfig.worldSize = bootstrap.worldSize;
                groupConfig.providerUniqueId = bootstrap.uniqueId;
            }
            else
            {
                groupConfig.worldRank = 0;
                groupConfig.worldSize = static_cast<int>(groupConfig.deviceIds.size());
            }

            auto const configureStatus = context.configureCollectives(groupConfig);
            if(configureStatus != tt::OpStatus::Success)
            {
                if(!bootstrap.enabled || bootstrap.worldRank == 0)
                {
                    std::cout << "Collective provider unavailable (status=" << static_cast<int>(configureStatus)
                              << "); skipping NCCL call.\n";
                }
                return 0;
            }

            std::size_t const participantCount = static_cast<std::size_t>(std::max(groupConfig.worldSize, 1));
            std::size_t const globalSamples = options.signalLength;
            if(globalSamples % participantCount != 0)
            {
                if(groupConfig.worldRank == 0)
                {
                    std::cout << "Signal length " << globalSamples
                              << " is not divisible by the participating rank count " << participantCount
                              << "; adjust --signal-length.\n";
                }
                return 0;
            }

            std::size_t const samplesPerRank = globalSamples / participantCount;
            if(samplesPerRank == 0)
            {
                if(groupConfig.worldRank == 0)
                {
                    std::cout << "Insufficient samples per rank (computed 0); increase --signal-length.\n";
                }
                return 0;
            }

            std::vector<std::complex<float>> localSamples;
            std::string chunkError;
            bool const chunkOk = loadSignalChunk(
                options,
                samplesPerRank,
                globalSamples,
                groupConfig.worldRank,
                participantCount,
                localSamples,
                chunkError);
            if(!chunkOk)
            {
                std::cerr << "Rank " << groupConfig.worldRank << " failed to load signal chunk: " << chunkError
                          << '\n';
                return 1;
            }

            auto const firstIndex = static_cast<std::size_t>(groupConfig.worldRank);
            auto const lastIndex = firstIndex + (samplesPerRank - 1) * participantCount;
            std::cout << "Rank " << groupConfig.worldRank << " owns " << samplesPerRank << " strided samples (indices "
                      << firstIndex << " to " << lastIndex << ").\n";
            if(options.signalFile && groupConfig.worldRank == 0)
            {
                std::cout << "Reading samples from file: " << *options.signalFile << '\n';
            }
            else if(!options.signalFile && groupConfig.worldRank == 0)
            {
                std::cout << "Generating synthetic multi-tone signal; provide --signal-file for real data.\n";
            }

            // Version 1 pipeline step 3: compute the naive local DFT on the host.
            auto const localSpectrum = computeLocalFft(localSamples);
            // Version 1 pipeline step 4: apply stride-dependent phases and lay out global contributions.
            auto contributions
                = buildRankContribution(localSpectrum, groupConfig.worldRank, participantCount, globalSamples);

            // Version 1 pipeline step 5: stage contributions in an alpaka tensor and move to the CUDA device.
            std::vector<float> interleaved(contributions.size() * 2U, 0.0f);
            for(std::size_t k = 0; k < contributions.size(); ++k)
            {
                interleaved[2 * k] = contributions[k].real();
                interleaved[2 * k + 1] = contributions[k].imag();
            }

            tt::Tensor1D<float, Device> spectralTensor(device, {interleaved.size()}, "fft-global-spectrum");
            auto* hostTensor = spectralTensor.hostData();
            std::copy(interleaved.begin(), interleaved.end(), hostTensor);
            spectralTensor.markHostModified();

            spectralTensor.ensureOnDevice(device, queue);
            auto& deviceBuffer = spectralTensor.deviceBuffer(device, queue);
            auto* devicePtr = alpaka::onHost::data(deviceBuffer);

            std::array<void*, 1> recvPtrs{static_cast<void*>(devicePtr)};
            std::array<void*, 1> streamPtrs{reinterpret_cast<void*>(alpaka::onHost::getNativeHandle(queue))};

            collective::MultiDeviceBuffers buffers{};
            buffers.recv = std::span<void*>{recvPtrs};
            buffers.streams = std::span<void*>{streamPtrs};
            buffers.inPlace = true;

            collective::AllReduceRequest request{};
            request.buffers = buffers;
            request.elementCount = spectralTensor.size();
            request.dataType = collective::DataType::Float32;
            request.reduceOp = collective::ReduceOp::Sum;

            // Version 1 pipeline step 6: use NCCL all-reduce on device memory to assemble the global FFT.
            auto const reduceStatus = context.collectiveAllReduce(request);
            if(reduceStatus != tt::OpStatus::Success)
            {
                std::cerr << "ncclAllReduce returned status=" << static_cast<int>(reduceStatus)
                          << "; cannot assemble global FFT.\n";
                return 1;
            }

            // Version 1 pipeline step 7: bring the merged spectrum back to host space for inspection.
            spectralTensor.markDeviceModified(device, queue);
            spectralTensor.toHost(device, queue);
            alpaka::onHost::wait(queue);

            auto const* reducedHost = spectralTensor.hostData();
            std::vector<std::complex<float>> globalSpectrum(globalSamples, std::complex<float>{0.0f, 0.0f});
            for(std::size_t k = 0; k < globalSamples; ++k)
            {
                globalSpectrum[k] = std::complex<float>{reducedHost[2 * k], reducedHost[2 * k + 1]};
            }

            printSpectrumPreview(groupConfig.worldRank, globalSpectrum, options.previewBins);

            if(groupConfig.worldRank == 0)
            {
                std::cout << "Verification settings: direct=" << (options.verifyDirect ? "on" : "off")
                          << ", reference=" << (options.referenceFftFile ? "path supplied" : "none")
                          << ", abs tol=" << options.verifyAbsTolerance << ", rel tol=" << options.verifyRelTolerance
                          << '\n';

                if(options.verifyDirect)
                {
                    std::cout << "Direct DFT precision: "
                              << (options.verifyPrecision == DirectVerifyPrecision::Float64 ? "float64" : "float32")
                              << "\n";
                }
                if(options.referenceFftFile)
                {
                    std::cout << "Reference FFT file: " << *options.referenceFftFile << '\n';
                }

                // Version 1 pipeline step 9: compare the distributed result against direct and file-based references.
                auto performVerification
                    = [&](std::vector<std::complex<float>> const& reference, std::string_view sourceLabel)
                {
                    std::span<std::complex<float> const> computedSpan{globalSpectrum.data(), globalSpectrum.size()};
                    std::span<std::complex<float> const> referenceSpan{reference.data(), reference.size()};
                    VerificationStats stats{};
                    bool const ok = compareSpectra(
                        computedSpan,
                        referenceSpan,
                        options.verifyAbsTolerance,
                        options.verifyRelTolerance,
                        stats);

                    if(ok)
                    {
                        std::cout << "Verification against " << sourceLabel
                                  << " passed (max abs diff=" << stats.maxAbsError
                                  << ", max rel diff=" << stats.maxRelError << ").\n";
                    }
                    else
                    {
                        std::cout << "Verification against " << sourceLabel << " failed (" << stats.failureCount
                                  << " bins over tolerance; max abs diff=" << stats.maxAbsError << " at bin "
                                  << stats.worstIndex << ", max rel diff=" << stats.maxRelError << ").\n";
                    }
                    return ok;
                };

                bool verificationRequested = options.verifyDirect || options.referenceFftFile.has_value();
                bool verificationSucceeded = true;
                bool verificationPerformed = false;

                if(options.referenceFftFile)
                {
                    std::vector<std::complex<float>> referenceSpectrum;
                    std::string referenceError;
                    if(loadReferenceSpectrumFromFile(
                           *options.referenceFftFile,
                           globalSamples,
                           referenceSpectrum,
                           referenceError))
                    {
                        verificationPerformed = true;
                        verificationSucceeded
                            = performVerification(referenceSpectrum, "reference FFT file") && verificationSucceeded;
                    }
                    else
                    {
                        std::cerr << "Unable to load reference FFT file: " << referenceError << '\n';
                        verificationSucceeded = false;
                    }
                }

                if(options.verifyDirect)
                {
                    if(globalSamples > (1U << 15))
                    {
                        std::cout << "Running naive O(n^2) verification for " << globalSamples
                                  << " samples; this may take a while.\n";
                    }

                    std::vector<std::complex<float>> fullSignal;
                    std::string fullSignalError;
                    if(loadFullSignalSequence(options, globalSamples, fullSignal, fullSignalError))
                    {
                        auto const directSpectrum = computeLocalFftWithPrecision(
                            std::span<std::complex<float> const>{fullSignal.data(), fullSignal.size()},
                            options.verifyPrecision);
                        verificationPerformed = true;
                        verificationSucceeded
                            = performVerification(directSpectrum, "direct naive DFT") && verificationSucceeded;
                    }
                    else
                    {
                        std::cerr << "Unable to assemble full input for verification: " << fullSignalError << '\n';
                        verificationSucceeded = false;
                    }
                }

                if(!verificationRequested)
                {
                    std::cout << "Verification disabled; enable with --verify-direct or provide --reference-fft.\n";
                }
                else if(!verificationPerformed)
                {
                    std::cout << "Verification could not be completed due to previous errors.\n";
                }
                else if(!verificationSucceeded)
                {
                    std::cout << "Distributed FFT verification failed.\n";
                }
                else
                {
                    std::cout << "Distributed FFT verification succeeded.\n";
                }

                std::cout << "Distributed FFT complete across " << participantCount
                          << " ranks using NCCL all-reduce.\n";
            }

            return 0;
        }
        else
        {
            if(!bootstrap.enabled || bootstrap.worldRank == 0)
            {
                std::cout << "Collective provider for NCCL is only available on CUDA executors; skipping.\n";
            }
            return 0;
        }
    }
} // namespace

int main(int argc, char** argv)
{
    auto bootstrap = collectiveFft::prepareBootstrap(argc, argv);
    auto options = collectiveFft::parseCommandLine(argc, argv);
    if((!bootstrap.enabled || bootstrap.worldRank == 0) && !options.warnings.empty())
    {
        for(auto const& warning : options.warnings)
        {
            std::cerr << "Option warning: " << warning << '\n';
        }
    }

    auto const result = alpaka::onHost::executeForEachIfHasDevice(
        [&bootstrap, &options](auto const& tag) { return runCollectiveFft(tag, bootstrap, options); },
        alpaka::onHost::allBackends(alpaka::onHost::enabledApis, alpaka::onHost::example::enabledExecutors));
    collectiveFft::finalizeBootstrap(bootstrap);
    return result;
}
