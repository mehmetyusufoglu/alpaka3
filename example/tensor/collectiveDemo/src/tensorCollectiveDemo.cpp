// Demo overview:
// - Shows how to instantiate a CleanTensorOpContext with collective support.
// - Configures a single-rank NCCL communicator targeting the first CUDA device.
// - Performs an in-place all-reduce to validate the NCCL provider wiring.
// - Falls back gracefully when CUDA or NCCL is unavailable on the current backend.
#include <alpaka/alpaka.hpp>
#include <alpaka/onHost/example/executors.hpp>
#include <alpaka/onHost/executeForEach.hpp>
#include <alpaka/onHost/interface.hpp>
#include <alpaka/tensor/core/TensorTypes.hpp>
#include <alpaka/tensor/providers/CleanTensorOpContext.hpp>
#include <alpaka/tensor/providers/EnabledVendorLibs.hpp>
#include <alpaka/tensor/providers/collective/CollectiveTypes.hpp>

#include <array>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <optional>
#include <span>
#include <string>
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
                    bootstrap.disableReason =
                        "ncclGetUniqueId failed with status=" + std::to_string(static_cast<int>(result));
                }
            }

            int const broadcastResult
                = MPI_Bcast(&id, static_cast<int>(sizeof(ncclUniqueId)), MPI_BYTE, 0, MPI_COMM_WORLD);
            if(broadcastResult != MPI_SUCCESS)
            {
                std::cerr << "MPI_Bcast failed while distributing NCCL unique ID; multi-rank NCCL disabled.\n";
                bootstrap.enabled = false;
                bootstrap.disableReason =
                    "MPI_Bcast failed while distributing NCCL unique ID (status="
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
            "OMPI_COMM_WORLD_LOCAL_RANK", "MPI_LOCALRANKID", "SLURM_LOCALID", "LOCAL_RANK"};

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
        bootstrap.disableReason =
            "Demo built without MPI support; rebuild with MPI to enable multi-rank collectives.";

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

    template<typename Tag>
    int runCollectiveDemo(Tag const& tag, MultiProcessBootstrap const& bootstrap)
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
            deviceCount = 1; // Defensive: avoid modulo by zero if selector misreports availability.
        }

        auto discoverLocalRank = [&]() -> std::optional<int>
        {
            if(bootstrap.localRank >= 0)
                return bootstrap.localRank;

            constexpr std::array<char const*, 4> envKeys{ "OMPI_COMM_WORLD_LOCAL_RANK",
                                                          "MPI_LOCALRANKID",
                                                          "SLURM_LOCALID",
                                                          "LOCAL_RANK" };
            for(auto const* key : envKeys)
            {
                if(auto envRank = parseEnvInt(key))
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
            std::cout << "\n=== Tensor Collective Demo ===\n";
            std::cout << "API: " << deviceSpec.getApi().getName() << '\n';
            std::cout << "Executor: " << alpaka::onHost::demangledName(exec) << '\n';
            if(bootstrap.enabled)
            {
                std::cout << "World size: " << bootstrap.worldSize << '\n';
                if(bootstrap.localRank >= 0)
                {
                    std::cout << "Local size: " << bootstrap.localSize << "\n";
                }
            }
            else if(bootstrap.worldSize > 1)
            {
                std::cout << "MPI detected " << bootstrap.worldSize
                          << " ranks but collectives stayed single-rank";
                if(!bootstrap.disableReason.empty())
                {
                    std::cout << " (" << bootstrap.disableReason << ")";
                }
                std::cout << "\n";
            }
        }

        if(!tt::EnabledVendorLibs::hasNCCL)
        {
            if(!bootstrap.enabled || bootstrap.worldRank == 0)
            {
                std::cout << "NCCL support not enabled in this build; skipping collective test.\n";
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

            constexpr std::size_t elementCount = 4;
            tt::Tensor1D<float, Device> values(device, {elementCount}, "collective-demo-values");

            auto* hostValues = values.hostData();
            for(std::size_t i = 0; i < elementCount; ++i)
                hostValues[i] = static_cast<float>(i + 1);
            values.markHostModified();

            values.ensureOnDevice(device, queue);
            auto& deviceBuffer = values.deviceBuffer(device, queue);
            auto* deviceValues = alpaka::onHost::data(deviceBuffer);

            std::array<void*, 1> recvPtrs{static_cast<void*>(deviceValues)};
            std::array<void*, 1> streamPtrs{reinterpret_cast<void*>(alpaka::onHost::getNativeHandle(queue))};

            collective::MultiDeviceBuffers buffers{};
            buffers.recv = std::span<void*>{recvPtrs};
            buffers.streams = std::span<void*>{streamPtrs};
            buffers.inPlace = true;

            collective::AllReduceRequest request{};
            request.buffers = buffers;
            request.elementCount = elementCount;
            request.dataType = collective::DataType::Float32;
            request.reduceOp = collective::ReduceOp::Sum;

            auto const reduceStatus = context.collectiveAllReduce(request);
            if(reduceStatus != tt::OpStatus::Success)
            {
                if(!bootstrap.enabled || bootstrap.worldRank == 0)
                {
                    std::cout << "ncclAllReduce invocation returned status=" << static_cast<int>(reduceStatus)
                              << "; skipping verification.\n";
                }
                return 0;
            }

            values.markDeviceModified(device, queue);
            values.toHost(device, queue);
            alpaka::onHost::wait(queue);

            std::cout << "Rank " << groupConfig.worldRank << " result:";
            for(std::size_t i = 0; i < elementCount; ++i)
                std::cout << ' ' << hostValues[i];
            if(bootstrap.enabled)
            {
                std::cout << "\n";
            }
            else
            {
                std::cout
                    << "\n(Note: With a single GPU the values remain unchanged; on multi-GPU systems they represent "
                       "the rank sum.)\n";
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
    auto const result = alpaka::onHost::executeForEachIfHasDevice(
        [&bootstrap](auto const& tag) { return runCollectiveDemo(tag, bootstrap); },
        alpaka::onHost::allBackends(alpaka::onHost::enabledApis, alpaka::onHost::example::enabledExecutors));
    finalizeBootstrap(bootstrap);
    return result;
}


