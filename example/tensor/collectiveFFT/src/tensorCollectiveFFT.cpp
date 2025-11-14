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
#include "collectiveFft/utilities.hpp"
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
#include <chrono>
#include <complex>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <thread>
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
    using collectiveFft::providerFftEnabled;
    using collectiveFft::providerFftRequired;
    using collectiveFft::VerificationStats;

    namespace detail = collectiveFft::detail;

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
            auto const preferredRank = detail::discoverLocalRank(bootstrap).value_or(bootstrap.worldRank);
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

            constexpr std::size_t maxDirectVerifySamples = 1U << 16; // 65536 bins keep O(N^2) reasonable
            constexpr std::size_t maxCufft1dLength = 1U << 26; // 67108864 bins is the cuFFT single-FFT limit on A100
            constexpr std::size_t maxDirectFallbackSamples = 1U << 18; // 262144 bins keeps CPU fallback bounded
            bool const directVerifyActive = options.verifyDirect && globalSamples <= maxDirectVerifySamples;
            if(options.verifyDirect && !directVerifyActive && groupConfig.worldRank == 0)
            {
                std::cout << "Skipping --verify-direct for signal length " << globalSamples << " (exceeds "
                          << maxDirectVerifySamples
                          << "); run with a smaller preview or offline verification if needed.\n";
            }

            bool const providerAvailable
                = tt::EnabledVendorLibs::hasCUFFT && context.supportsOperation(tt::OpType::FFT);
            bool const sizeWithinProvider
                = samplesPerRank <= static_cast<std::size_t>(std::numeric_limits<int>::max());
            bool const lengthWithinCufftLimit = samplesPerRank <= maxCufft1dLength;
            bool const providerUsable = providerAvailable && sizeWithinProvider && lengthWithinCufftLimit;

            if(providerFftRequired(options) && !providerUsable)
            {
                if(groupConfig.worldRank == 0)
                {
                    if(!providerAvailable)
                    {
                        std::cerr << "cuFFT provider required (--force-provider-fft) but not available in this "
                                     "configuration."
                                  << '\n';
                    }
                    else
                    {
                        std::cerr << "cuFFT provider required (--force-provider-fft) but local transform length "
                                  << samplesPerRank << " exceeds the single-FFT cuFFT limit (~" << maxCufft1dLength
                                  << "); increase the MPI rank count or implement tiled FFTs.\n";
                    }
                }
                return 1;
            }

            if(providerFftEnabled(options) && !providerUsable && groupConfig.worldRank == 0)
            {
                if(!providerAvailable)
                {
                    std::cout << "cuFFT provider unavailable; defaulting to host-side DFT for local spectra." << '\n';
                }
                else if(!sizeWithinProvider)
                {
                    std::cout << "Local FFT length " << samplesPerRank
                              << " exceeds cuFFT limits; defaulting to host-side DFT." << '\n';
                }
                else
                {
                    std::cout << "Local FFT length " << samplesPerRank << " exceeds the single-FFT cuFFT limit (~"
                              << maxCufft1dLength << "); defaulting to host-side DFT." << '\n';
                }
            }

            if(!providerFftEnabled(options) && providerAvailable && groupConfig.worldRank == 0)
            {
                std::cout << "cuFFT provider disabled via --disable-provider-fft; using host-side DFT." << '\n';
            }

            bool const useProviderFft = providerFftEnabled(options) && providerUsable;

            if(useProviderFft && groupConfig.worldRank == 0)
            {
                std::cout << "cuFFT provider active for local spectra on each rank." << '\n';
            }

            std::vector<std::complex<float>> localSamples;
            std::string chunkError;
            // Stage 1: pull the strided slice of the global signal that belongs to this rank
            // into host memory (either from file or from the synthetic generator).
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
            auto const lastIndex = firstIndex + (samplesPerRank > 0 ? (samplesPerRank - 1) * participantCount : 0);
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

            if(!localSamples.empty())
            {
                double sampleMagnitudeSum = 0.0;
                float sampleMagnitudeMax = 0.0f;
                for(auto const& value : localSamples)
                {
                    float const magnitude = std::abs(value);
                    sampleMagnitudeSum += static_cast<double>(magnitude);
                    sampleMagnitudeMax = std::max(sampleMagnitudeMax, magnitude);
                }
                double const magnitudeMean = sampleMagnitudeSum / static_cast<double>(localSamples.size());
                std::cout << "Rank " << groupConfig.worldRank << " local samples stats: max |x|=" << sampleMagnitudeMax
                          << ", mean |x|=" << magnitudeMean << '\n';

                std::size_t const previewCount = std::min<std::size_t>(8, localSamples.size());
                std::cout << "Rank " << groupConfig.worldRank << " sample preview:";
                for(std::size_t idx = 0; idx < previewCount; ++idx)
                {
                    std::cout << ' ' << idx << ':' << localSamples[idx];
                }
                if(localSamples.size() > previewCount)
                {
                    std::cout << " ...";
                }
                std::cout << '\n';
            }

            std::vector<std::complex<float>> localSpectrum;

            if(useProviderFft)
            {
                // if(groupConfig.worldRank > 0)
                // {
                //     auto const delay = std::chrono::milliseconds(100 * groupConfig.worldRank);
                //     std::this_thread::sleep_for(delay);
                // }


                std::cout << "[Rank " << groupConfig.worldRank
                          << "] cuFFT context=" << static_cast<void const*>(&context) << " device=" << deviceId
                          << " samples=" << samplesPerRank << '\n';


                // Stage 2: allocate device resident tensors that map 1:1 onto the
                // cuFFT input/output buffers managed by CleanTensorOpContext.
                tt::Tensor1D<std::complex<float>, Device> deviceInput(device, {samplesPerRank}, "fft-local-input");
                tt::Tensor1D<std::complex<float>, Device> deviceOutput(device, {samplesPerRank}, "fft-local-output");

                auto* hostInputPtr = deviceInput.hostData();
                std::copy(localSamples.begin(), localSamples.end(), hostInputPtr);
                deviceInput.markHostModified();

                std::cout << "[Rank " << groupConfig.worldRank
                          << "] deviceInput host ptr=" << static_cast<void*>(hostInputPtr)
                          << ", sample[0]=" << (samplesPerRank > 0 ? hostInputPtr[0] : std::complex<float>{})
                          << ", sample[1]=" << (samplesPerRank > 1 ? hostInputPtr[1] : std::complex<float>{}) << '\n';

                if(groupConfig.worldRank == 0 && !localSamples.empty())
                {
                    std::cout << "[Rank 0] first input sample before upload=" << hostInputPtr[0] << '\n';
                }

                deviceInput.ensureOnDevice(device, queue);
                alpaka::onHost::wait(queue);

                {
                    auto& inputDeviceBuffer = deviceInput.deviceBuffer(device, queue);
                    auto* devicePtr = alpaka::onHost::data(inputDeviceBuffer);
                    std::cout << "[Rank " << groupConfig.worldRank
                              << "] deviceInput device ptr=" << static_cast<void*>(devicePtr) << " uploaded" << '\n';
                }

                deviceOutput.ensureOnDevice(device, queue);
                alpaka::onHost::wait(queue);

                {
                    auto& outputDeviceBuffer = deviceOutput.deviceBuffer(device, queue);
                    auto* devicePtr = alpaka::onHost::data(outputDeviceBuffer);
                    std::cout << "[Rank " << groupConfig.worldRank
                              << "] deviceOutput device ptr=" << static_cast<void*>(devicePtr) << " staged" << '\n';
                }

                tt::ops::FftParams fftParams{};
                fftParams.rank = 1;
                fftParams.lengths[0] = samplesPerRank;
                fftParams.batch = 1;
                fftParams.transformType = tt::ops::FftTransformType::ComplexToComplex;
                fftParams.direction = tt::ops::FftDirection::Forward;
                fftParams.inPlace = false;

                bool fftFailed = false;
                try
                {
                    // Stage 3: ask the CleanTensorOpContext to execute its FFT provider.
                    // On CUDA this drops into the cuFFT plan cached inside the context,
                    // writing frequency-domain coefficients into deviceOutput.
                    context.fft(deviceInput, deviceOutput, fftParams);
                    std::cout << "[Rank " << groupConfig.worldRank << "] cuFFT dispatched" << '\n';
                    deviceOutput.markDeviceModified(device, queue);
                    alpaka::onHost::wait(queue);

                    deviceOutput.toHost(device, queue);
                    alpaka::onHost::wait(queue);

                    auto const* hostOutputPtr = deviceOutput.hostData();
                    // Stage 4: capture the cuFFT spectrum back on the host so the
                    // subsequent tiled reduction can work in CPU memory.
                    std::cout << "[Rank " << groupConfig.worldRank
                              << "] deviceOutput host ptr=" << static_cast<void const*>(hostOutputPtr)
                              << " post-download" << '\n';
                    std::vector<std::complex<float>> deviceSpectrum(hostOutputPtr, hostOutputPtr + samplesPerRank);

                    bool const spectrumLooksZero = std::all_of(
                        deviceSpectrum.begin(),
                        deviceSpectrum.begin() + std::min<std::size_t>(deviceSpectrum.size(), 16),
                        [](std::complex<float> const& value) { return std::abs(value) < 1.0e-6f; });

                    if(groupConfig.worldRank == 0)
                    {
                        std::cout << "[Rank 0] first cuFFT output sample="
                                  << (deviceSpectrum.empty() ? std::complex<float>{} : deviceSpectrum.front())
                                  << " (zero spectrum check=" << (spectrumLooksZero ? "yes" : "no") << ")\n";
                    }

                    if(!deviceSpectrum.empty())
                    {
                        std::cout << "[Rank " << groupConfig.worldRank
                                  << "] cuFFT output preview: 0:" << deviceSpectrum[0];
                        std::size_t const outputPreview = std::min<std::size_t>(4, deviceSpectrum.size());
                        for(std::size_t idx = 1; idx < outputPreview; ++idx)
                        {
                            std::cout << ' ' << idx << ':' << deviceSpectrum[idx];
                        }
                        if(deviceSpectrum.size() > outputPreview)
                        {
                            std::cout << " ...";
                        }
                        std::cout << '\n';
                    }

                    if(spectrumLooksZero)
                    {
                        std::cerr << "Rank " << groupConfig.worldRank
                                  << " cuFFT spectrum appears zero; falling back to host DFT.\n";
                        fftFailed = true;
                    }
                    else
                    {
                        // Persist fft() results back on the host for later phase correction and reduction.
                        localSpectrum = std::move(deviceSpectrum);
                    }
                }
                catch(std::exception const& e)
                {
                    std::cerr << "Rank " << groupConfig.worldRank << " cuFFT failed: " << e.what()
                              << "; falling back to host DFT.\n";
                    fftFailed = true;
                }

                if(fftFailed)
                {
                    if(samplesPerRank > maxDirectFallbackSamples)
                    {
                        std::cerr << "Rank " << groupConfig.worldRank << " fallback disabled: O(N^2) DFT for "
                                  << samplesPerRank << " samples would exhaust memory/time. Aborting.\n";
                        return 1;
                    }
                    localSpectrum = computeLocalFft(localSamples);
                }
            }
            else
            {
                if(samplesPerRank > maxDirectFallbackSamples)
                {
                    if(groupConfig.worldRank == 0)
                    {
                        std::cerr << "CPU FFT fallback unavailable for " << samplesPerRank
                                  << " samples per rank; enable cuFFT or add more ranks." << '\n';
                    }
                    return 1;
                }
                localSpectrum = computeLocalFft(localSamples);
            }

            if(!localSpectrum.empty())
            {
                double magnitudeSum = 0.0;
                float magnitudeMax = 0.0f;
                for(auto const& value : localSpectrum)
                {
                    float const magnitude = std::abs(value);
                    magnitudeSum += static_cast<double>(magnitude);
                    magnitudeMax = std::max(magnitudeMax, magnitude);
                }
                double const magnitudeMean = magnitudeSum / static_cast<double>(localSpectrum.size());
                std::cout << "Rank " << groupConfig.worldRank << " local spectrum stats: max |X|=" << magnitudeMax
                          << ", mean |X|=" << magnitudeMean << '\n';

                if(groupConfig.worldRank == 0)
                {
                    std::cout << "Rank 0 local spectrum preview:";
                    std::size_t const previewBins = std::min<std::size_t>(8, localSpectrum.size());
                    for(std::size_t idx = 0; idx < previewBins; ++idx)
                    {
                        std::cout << " " << idx << ":" << localSpectrum[idx];
                    }
                    std::cout << '\n';
                }
            }
            /**
             * Version 2 pipeline step 4: stream the global spectrum back in fixed-size tiles rather than
             * allocating an N-length buffer per rank. Each pass uploads only tileCapacity bins to the GPU,
             * runs the NCCL all-reduce on that slice, and copies the reduced result to host. The reuse of
             * those buffers keeps memory bounded regardless of the global FFT length and lets very large
             * signals complete without exhausting device or host space.
             */
            std::cout << "Rank " << groupConfig.worldRank
                      << " assembling contributions with tiled all-reduce (tile size selected for memory bounds).\n";

            constexpr std::size_t defaultTile = 1U << 18; // 262144 bins (~2 MB of complex float data)
            std::size_t const tileCapacity
                = std::max<std::size_t>(1, std::min<std::size_t>(defaultTile, globalSamples));
            std::vector<std::complex<float>> tileContributions(tileCapacity, std::complex<float>{0.0f, 0.0f});
            bool const collectFullSpectrum = directVerifyActive || options.referenceFftFile.has_value();
            std::vector<std::complex<float>> globalSpectrum;
            if(collectFullSpectrum)
            {
                globalSpectrum.assign(globalSamples, std::complex<float>{0.0f, 0.0f});
            }
            std::size_t const previewLimit = std::min<std::size_t>(options.previewBins, globalSamples);
            std::vector<std::complex<float>> previewSpectrum;
            if(!collectFullSpectrum && previewLimit > 0U)
            {
                previewSpectrum.assign(previewLimit, std::complex<float>{0.0f, 0.0f});
            }
            std::size_t previewWritten = 0U;

            tt::Tensor1D<float, Device> spectralTensor(device, {tileCapacity * 2U}, "fft-global-spectrum-tile");
            auto* hostTensor = spectralTensor.hostData();

            // Allocate device storage once; per-tile uploads will refresh the active prefix.
            spectralTensor.ensureOnDevice(device, queue);
            alpaka::onHost::wait(queue);
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
            request.dataType = collective::DataType::Float32;
            request.reduceOp = collective::ReduceOp::Sum;

            for(std::size_t tileOffset = 0; tileOffset < globalSamples; tileOffset += tileCapacity)
            {
                std::size_t const tileLength = std::min<std::size_t>(tileCapacity, globalSamples - tileOffset);
                auto tileSpan = std::span<std::complex<float>>{tileContributions.data(), tileLength};

                detail::buildContributionTile(
                    std::span<std::complex<float> const>{localSpectrum.data(), localSpectrum.size()},
                    groupConfig.worldRank,
                    participantCount,
                    globalSamples,
                    tileOffset,
                    tileSpan);

                if(tileOffset == 0 && !tileSpan.empty())
                {
                    std::size_t const stridedPreview = std::min<std::size_t>(8, samplesPerRank);
                    std::cout << "Rank " << groupConfig.worldRank << " contribution preview (bins rank + m*stride):";
                    for(std::size_t idx = 0; idx < stridedPreview; ++idx)
                    {
                        std::size_t const globalIndex
                            = static_cast<std::size_t>(groupConfig.worldRank) + idx * participantCount;
                        if(globalIndex >= globalSamples)
                            break;
                        std::size_t const localOffset = globalIndex - tileOffset;
                        if(localOffset >= tileSpan.size())
                            break;
                        std::cout << ' ' << globalIndex << ':' << tileSpan[localOffset];
                    }
                    if(samplesPerRank > stridedPreview)
                    {
                        std::cout << " ...";
                    }
                    std::cout << '\n';
                }

                for(std::size_t i = 0; i < tileLength; ++i)
                {
                    hostTensor[2 * i] = tileSpan[i].real();
                    hostTensor[2 * i + 1] = tileSpan[i].imag();
                }

                spectralTensor.markHostModified();
                spectralTensor.ensureOnDevice(device, queue);
                alpaka::onHost::wait(queue);

                request.elementCount = tileLength * 2U;
                auto const reduceStatus = context.collectiveAllReduce(request);
                if(reduceStatus != tt::OpStatus::Success)
                {
                    std::cerr << "ncclAllReduce returned status=" << static_cast<int>(reduceStatus)
                              << "; cannot assemble global FFT.\n";
                    return 1;
                }

                spectralTensor.markDeviceModified(device, queue);
                spectralTensor.toHost(device, queue);
                alpaka::onHost::wait(queue);

                auto const* reducedHost = spectralTensor.hostData();
                if(collectFullSpectrum)
                {
                    for(std::size_t i = 0; i < tileLength; ++i)
                    {
                        globalSpectrum[tileOffset + i]
                            = std::complex<float>{reducedHost[2 * i], reducedHost[2 * i + 1]};
                    }
                }
                else if(!previewSpectrum.empty() && previewWritten < previewSpectrum.size())
                {
                    std::size_t const toCopy
                        = std::min<std::size_t>(previewSpectrum.size() - previewWritten, tileLength);
                    for(std::size_t i = 0; i < toCopy; ++i)
                    {
                        previewSpectrum[previewWritten + i]
                            = std::complex<float>{reducedHost[2 * i], reducedHost[2 * i + 1]};
                    }
                    previewWritten += toCopy;
                }
            }

            if(collectFullSpectrum)
            {
                detail::printSpectrumPreview(groupConfig.worldRank, globalSpectrum, options.previewBins);
            }
            else if(groupConfig.worldRank == 0 && !previewSpectrum.empty())
            {
                std::cout << "First " << previewSpectrum.size()
                          << " frequency bins (partial preview, streaming mode):\n";
                for(std::size_t k = 0; k < previewSpectrum.size(); ++k)
                {
                    auto const value = previewSpectrum[k];
                    std::cout << "  k=" << k << " : " << value.real() << "  " << value.imag()
                              << "i  |X|=" << std::abs(value) << '\n';
                }
                std::cout << "Dominant bin search skipped (full spectrum not retained).\n";
            }

            if(groupConfig.worldRank == 0)
            {
                std::cout << "Verification settings: direct=" << (directVerifyActive ? "on" : "off")
                          << ", reference=" << (options.referenceFftFile ? "path supplied" : "none")
                          << ", abs tol=" << options.verifyAbsTolerance << ", rel tol=" << options.verifyRelTolerance
                          << '\n';

                if(directVerifyActive)
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

                bool verificationRequested = directVerifyActive || options.referenceFftFile.has_value();
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

                if(directVerifyActive)
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
