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
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <numbers>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#ifdef ALPAKA_TENSOR_COLLECTIVE_DEMO_HAS_MPI
#    include <mpi.h>
#endif

#ifdef ALPAKA_HAS_NCCL
#    include <nccl.h>
#endif

namespace tt = alpaka::tensor;
namespace collective = alpaka::tensor::collective;

namespace
{
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

    template<std::size_t N>
    [[nodiscard]] std::optional<int> parseFirstEnvInt(std::array<char const*, N> const& envNames)
    {
        for(auto const* name : envNames)
        {
            if(auto parsed = parseEnvInt(name))
                return parsed;
        }
        return std::nullopt;
    }

    struct MultiProcessBootstrap
    {
        bool enabled = false;
        bool finalizeMpi = false;
        int worldRank = 0;
        int worldSize = 1;
        int localRank = -1;
        int localSize = -1;
        std::string disableReason{};
        std::vector<std::byte> uniqueId{};
    };

#ifdef ALPAKA_TENSOR_COLLECTIVE_DEMO_HAS_MPI
    MultiProcessBootstrap prepareBootstrap(int& argc, char**& argv)
    {
        MultiProcessBootstrap bootstrap{};
        int initialized = 0;
        auto const initQueryStatus = MPI_Initialized(&initialized);
        if(initQueryStatus != MPI_SUCCESS)
        {
            bootstrap.disableReason = "MPI_Initialized failed with status=" + std::to_string(initQueryStatus);
            return bootstrap;
        }

        if(!initialized)
        {
            int provided = MPI_THREAD_SINGLE;
            auto const initStatus = MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
            if(initStatus != MPI_SUCCESS)
            {
                bootstrap.disableReason = "MPI_Init_thread failed with status=" + std::to_string(initStatus);
                return bootstrap;
            }
            bootstrap.finalizeMpi = true;
        }

        auto const commSizeStatus = MPI_Comm_size(MPI_COMM_WORLD, &bootstrap.worldSize);
        if(commSizeStatus != MPI_SUCCESS)
        {
            bootstrap.disableReason = "MPI_Comm_size failed with status=" + std::to_string(commSizeStatus);
            bootstrap.worldSize = 1;
        }

        auto const commRankStatus = MPI_Comm_rank(MPI_COMM_WORLD, &bootstrap.worldRank);
        if(commRankStatus != MPI_SUCCESS)
        {
            bootstrap.disableReason = "MPI_Comm_rank failed with status=" + std::to_string(commRankStatus);
            bootstrap.worldRank = 0;
        }
        bootstrap.enabled = bootstrap.worldSize > 1;

        if(bootstrap.worldSize > 1)
        {
            MPI_Comm localComm = MPI_COMM_NULL;
            auto const splitStatus = MPI_Comm_split_type(
                MPI_COMM_WORLD,
                MPI_COMM_TYPE_SHARED,
                bootstrap.worldRank,
                MPI_INFO_NULL,
                &localComm);
            if(splitStatus == MPI_SUCCESS && localComm != MPI_COMM_NULL)
            {
                MPI_Comm_rank(localComm, &bootstrap.localRank);
                MPI_Comm_size(localComm, &bootstrap.localSize);
                MPI_Comm_free(&localComm);
            }
            else
            {
                bootstrap.localRank = bootstrap.worldRank;
                bootstrap.localSize = bootstrap.worldSize;
            }
        }
        else
        {
            bootstrap.localRank = 0;
            bootstrap.localSize = 1;
        }

#    ifdef ALPAKA_HAS_NCCL
        if(bootstrap.enabled)
        {
            ncclUniqueId id{};
            if(bootstrap.worldRank == 0)
            {
                auto const result = ncclGetUniqueId(&id);
                if(result != ncclSuccess)
                {
                    std::cerr << "ncclGetUniqueId failed with status=" << static_cast<int>(result)
                              << "; multi-rank NCCL disabled.\n";
                    bootstrap.enabled = false;
                    bootstrap.disableReason
                        = "ncclGetUniqueId failed with status=" + std::to_string(static_cast<int>(result));
                }
            }

            int const broadcastResult
                = MPI_Bcast(&id, static_cast<int>(sizeof(ncclUniqueId)), MPI_BYTE, 0, MPI_COMM_WORLD);
            if(broadcastResult != MPI_SUCCESS)
            {
                std::cerr << "MPI_Bcast failed while distributing NCCL unique ID; multi-rank NCCL disabled.\n";
                bootstrap.enabled = false;
                bootstrap.disableReason = "MPI_Bcast failed while distributing NCCL unique ID (status="
                                          + std::to_string(broadcastResult) + ")";
            }
            int enabledFlag = bootstrap.enabled ? 1 : 0;
            MPI_Bcast(&enabledFlag, 1, MPI_INT, 0, MPI_COMM_WORLD);
            bootstrap.enabled = enabledFlag != 0;

            if(bootstrap.enabled)
            {
                bootstrap.uniqueId.resize(sizeof(ncclUniqueId));
                std::memcpy(bootstrap.uniqueId.data(), &id, sizeof(ncclUniqueId));
            }
            else
            {
                bootstrap.uniqueId.clear();
            }
        }
#    else
        if(bootstrap.enabled)
        {
            bootstrap.disableReason = "NCCL support not enabled in this build.";
            bootstrap.enabled = false;
            bootstrap.uniqueId.clear();
        }
#    endif

        if(bootstrap.worldSize > 1)
        {
            int enabledFlag = bootstrap.enabled ? 1 : 0;
            MPI_Bcast(&enabledFlag, 1, MPI_INT, 0, MPI_COMM_WORLD);
            bootstrap.enabled = enabledFlag != 0;

            int reasonLength = static_cast<int>(bootstrap.disableReason.size());
            MPI_Bcast(&reasonLength, 1, MPI_INT, 0, MPI_COMM_WORLD);
            if(bootstrap.worldRank != 0)
            {
                bootstrap.disableReason.resize(static_cast<std::size_t>(reasonLength));
            }
            if(reasonLength > 0)
            {
                MPI_Bcast(bootstrap.disableReason.data(), reasonLength, MPI_CHAR, 0, MPI_COMM_WORLD);
            }
        }

        return bootstrap;
    }
#else
    MultiProcessBootstrap prepareBootstrap(int&, char**&)
    {
        MultiProcessBootstrap bootstrap{};
        static constexpr std::array<char const*, 2> worldSizeKeys{"OMPI_COMM_WORLD_SIZE", "WORLD_SIZE"};
        static constexpr std::array<char const*, 2> worldRankKeys{"OMPI_COMM_WORLD_RANK", "RANK"};
        static constexpr std::array<char const*, 4> localRankKeys{
            "OMPI_COMM_WORLD_LOCAL_RANK",
            "MPI_LOCALRANKID",
            "SLURM_LOCALID",
            "LOCAL_RANK"};

        if(auto worldSize = parseFirstEnvInt(worldSizeKeys))
            bootstrap.worldSize = *worldSize;
        if(auto worldRank = parseFirstEnvInt(worldRankKeys))
            bootstrap.worldRank = *worldRank;
        if(auto localRank = parseFirstEnvInt(localRankKeys))
            bootstrap.localRank = *localRank;
        else
            bootstrap.localRank = bootstrap.worldRank;

        bootstrap.localSize = bootstrap.worldSize;
        bootstrap.enabled = false;
        bootstrap.disableReason = "Demo built without MPI support; rebuild with MPI to enable multi-rank collectives.";

        return bootstrap;
    }
#endif

    void finalizeBootstrap(MultiProcessBootstrap& bootstrap)
    {
#ifdef ALPAKA_TENSOR_COLLECTIVE_DEMO_HAS_MPI
        if(bootstrap.finalizeMpi)
        {
            MPI_Finalize();
            bootstrap.finalizeMpi = false;
        }
#else
        static_cast<void>(bootstrap);
#endif
    }

    struct CommandLineOptions
    {
        std::size_t signalLength = 16384;
        std::optional<std::string> signalFile{};
        std::size_t previewBins = 12;
        std::vector<std::string> warnings{};
    };

    CommandLineOptions parseCommandLine(int argc, char** argv)
    {
        CommandLineOptions options{};
        constexpr std::string_view lengthPrefix{"--signal-length="};
        constexpr std::string_view filePrefix{"--signal-file="};
        constexpr std::string_view previewPrefix{"--preview-bins="};

        for(int i = 1; i < argc; ++i)
        {
            std::string_view arg(argv[i]);
            if(arg.rfind(lengthPrefix, 0) == 0)
            {
                std::string value(arg.substr(lengthPrefix.size()));
                char* end = nullptr;
                unsigned long long parsed = std::strtoull(value.c_str(), &end, 10);
                if(end == value.c_str() || parsed == 0ULL)
                {
                    options.warnings.emplace_back(
                        "Ignoring invalid --signal-length value '" + value + "'; keeping default.");
                }
                else
                {
                    options.signalLength = static_cast<std::size_t>(parsed);
                }
            }
            else if(arg.rfind(filePrefix, 0) == 0)
            {
                std::string path(arg.substr(filePrefix.size()));
                if(path.empty())
                {
                    options.warnings.emplace_back("Ignoring empty --signal-file argument.");
                }
                else
                {
                    options.signalFile = std::move(path);
                }
            }
            else if(arg.rfind(previewPrefix, 0) == 0)
            {
                std::string value(arg.substr(previewPrefix.size()));
                char* end = nullptr;
                unsigned long long parsed = std::strtoull(value.c_str(), &end, 10);
                if(end == value.c_str() || parsed == 0ULL)
                {
                    options.warnings.emplace_back(
                        "Ignoring invalid --preview-bins value '" + value + "'; keeping default.");
                }
                else
                {
                    options.previewBins = static_cast<std::size_t>(parsed);
                }
            }
        }

        if(options.signalLength == 0)
            options.signalLength = 1;
        if(options.previewBins == 0)
            options.previewBins = 1;
        return options;
    }

    float syntheticSample(std::size_t globalIndex, std::size_t totalLength)
    {
        constexpr float twoPi = 2.0f * std::numbers::pi_v<float>;
        float position = (totalLength > 0) ? static_cast<float>(globalIndex) / static_cast<float>(totalLength) : 0.0f;
        float base = std::sin(twoPi * 3.0f * position);
        float overtone = 0.35f * std::sin(twoPi * 17.0f * position + 0.6f);
        float envelope = 0.5f * std::cos(twoPi * 1.0f * position);
        return base + overtone + envelope;
    }

    bool loadSignalChunk(
        CommandLineOptions const& options,
        std::size_t samplesPerRank,
        std::size_t totalSamples,
        int worldRank,
        std::size_t worldSize,
        std::vector<std::complex<float>>& chunk,
        std::string& errorMessage)
    {
        chunk.assign(samplesPerRank, std::complex<float>{0.0f, 0.0f});
        std::size_t const stride = worldSize;

        if(options.signalFile)
        {
            std::ifstream input(options.signalFile->c_str(), std::ios::binary);
            if(!input)
            {
                errorMessage = "Failed to open signal file '" + *options.signalFile + "'.";
                return false;
            }

            for(std::size_t m = 0; m < samplesPerRank; ++m)
            {
                std::size_t const globalIndex = stride * m + static_cast<std::size_t>(worldRank);
                if(globalIndex >= totalSamples)
                    break;

                std::streamoff const offset
                    = static_cast<std::streamoff>(globalIndex) * static_cast<std::streamoff>(sizeof(float));
                input.seekg(offset, std::ios::beg);
                if(!input.good())
                {
                    errorMessage = "seekg failed at byte offset " + std::to_string(offset);
                    return false;
                }

                float sample = 0.0f;
                input.read(reinterpret_cast<char*>(&sample), sizeof(float));
                if(!input)
                {
                    errorMessage = "Failed to read sample index " + std::to_string(globalIndex);
                    return false;
                }

                chunk[m] = std::complex<float>{sample, 0.0f};
            }
        }
        else
        {
            for(std::size_t m = 0; m < samplesPerRank; ++m)
            {
                std::size_t const globalIndex = stride * m + static_cast<std::size_t>(worldRank);
                if(globalIndex >= totalSamples)
                    break;
                chunk[m] = std::complex<float>{syntheticSample(globalIndex, totalSamples), 0.0f};
            }
        }

        return true;
    }

    // Naive O(n^2) DFT of the rank's strided sample sequence; good enough for demonstration sizes.
    std::vector<std::complex<float>> computeLocalFft(std::span<std::complex<float> const> samples)
    {
        std::size_t const sampleCount = samples.size();
        std::vector<std::complex<float>> spectrum(sampleCount, std::complex<float>{0.0f, 0.0f});
        if(sampleCount == 0)
            return spectrum;

        constexpr float twoPi = 2.0f * std::numbers::pi_v<float>;
        for(std::size_t k = 0; k < sampleCount; ++k)
        {
            std::complex<float> sum{0.0f, 0.0f};
            for(std::size_t n = 0; n < sampleCount; ++n)
            {
                float angle = -twoPi * static_cast<float>(n * k) / static_cast<float>(sampleCount);
                sum += samples[n] * std::complex<float>{std::cos(angle), std::sin(angle)};
            }
            spectrum[k] = sum;
        }

        return spectrum;
    }

    // Compose the global spectrum contribution for this rank by applying offset-dependent phase factors.
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

        auto discoverLocalRank = [&]() -> std::optional<int>
        {
            if(bootstrap.localRank >= 0)
                return bootstrap.localRank;

            constexpr std::array<char const*, 4> envKeys{
                "OMPI_COMM_WORLD_LOCAL_RANK",
                "MPI_LOCALRANKID",
                "SLURM_LOCALID",
                "LOCAL_RANK"};
            for(auto const* key : envKeys)
            {
                if(auto const envRank = parseEnvInt(key))
                    return envRank;
            }
            return std::nullopt;
        };

        int deviceId = 0;
        if(deviceCount > 0)
        {
            auto const preferredRank = discoverLocalRank().value_or(bootstrap.worldRank);
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

            auto const localSpectrum = computeLocalFft(localSamples);
            auto contributions
                = buildRankContribution(localSpectrum, groupConfig.worldRank, participantCount, globalSamples);

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
            std::vector<std::complex<float>> globalSpectrum(globalSamples, std::complex<float>{0.0f, 0.0f});
            for(std::size_t k = 0; k < globalSamples; ++k)
            {
                globalSpectrum[k] = std::complex<float>{reducedHost[2 * k], reducedHost[2 * k + 1]};
            }

            printSpectrumPreview(groupConfig.worldRank, globalSpectrum, options.previewBins);

            if(groupConfig.worldRank == 0)
            {
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
    auto bootstrap = prepareBootstrap(argc, argv);
    auto options = parseCommandLine(argc, argv);
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
    finalizeBootstrap(bootstrap);
    return result;
}
