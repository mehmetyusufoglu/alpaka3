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
#include <cstring>
#include <iostream>
#include <span>
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
    struct MultiProcessBootstrap
    {
        bool enabled = false;
        bool finalizeMpi = false;
        int worldRank = 0;
        int worldSize = 1;
        std::vector<std::byte> uniqueId{};
    };

#ifdef ALPAKA_TENSOR_COLLECTIVE_DEMO_HAS_MPI
    MultiProcessBootstrap prepareBootstrap(int& argc, char**& argv)
    {
        MultiProcessBootstrap bootstrap{};
        int initialized = 0;
        MPI_Initialized(&initialized);
        if(!initialized)
        {
            int provided = MPI_THREAD_SINGLE;
            MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
            bootstrap.finalizeMpi = true;
        }

        MPI_Comm_size(MPI_COMM_WORLD, &bootstrap.worldSize);
        MPI_Comm_rank(MPI_COMM_WORLD, &bootstrap.worldRank);
        bootstrap.enabled = bootstrap.worldSize > 1;

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
                }
            }

            int const broadcastResult
                = MPI_Bcast(&id, static_cast<int>(sizeof(ncclUniqueId)), MPI_BYTE, 0, MPI_COMM_WORLD);
            if(broadcastResult != MPI_SUCCESS)
            {
                std::cerr << "MPI_Bcast failed while distributing NCCL unique ID; multi-rank NCCL disabled.\n";
                bootstrap.enabled = false;
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
#    endif

        return bootstrap;
    }
#else
    MultiProcessBootstrap prepareBootstrap(int&, char**&)
    {
        return {};
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

        auto device = selector.makeDevice(0);
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

            collective::GroupConfig groupConfig{};
            groupConfig.deviceIds.push_back(0); // assumes CUDA_VISIBLE_DEVICES pins ranks appropriately
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
