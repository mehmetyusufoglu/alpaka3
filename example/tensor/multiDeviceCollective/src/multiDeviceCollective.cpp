// Multi-device NCCL/RCCL collective example
// SPDX-License-Identifier: MPL-2.0

#include <alpaka/alpaka.hpp>
#include <alpaka/onHost/example/executors.hpp>
#include <alpaka/onHost/executeForEach.hpp>
#include <alpaka/onHost/interface.hpp>
#include <alpaka/tensor/ops/CollectiveOps.hpp>
#include <alpaka/tensor/providers/ICollectiveProvider.hpp>
#include <alpaka/tensor/providers/NCCLProvider.hpp>
#include <alpaka/tensor/providers/RCCLProvider.hpp>

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <iostream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#include <thread>
#include <future>

namespace at = alpaka::tensor;
namespace ops = alpaka::tensor::ops;

namespace
{
    template<typename Diagnostics>
    void reportDiagnostics(Diagnostics const& diag)
    {
        std::cout << "[multi-collective] detected GPUs: " << diag.deviceCount;
        if(diag.activeDevice >= 0)
        {
            std::cout << " (active=" << diag.activeDevice;
            if(!diag.activeDeviceName.empty())
            {
                std::cout << ", name=\"" << diag.activeDeviceName << "\"";
            }
            std::cout << ')';
        }
        std::cout << '\n';

        if(!diag.deviceNames.empty())
        {
            for(std::size_t idx = 0; idx < diag.deviceNames.size(); ++idx)
            {
                std::cout << "[multi-collective]   gpu[" << idx << "]: " << diag.deviceNames[idx] << '\n';
            }
        }

        if(diag.deviceCount > 1 && diag.peerAccess.size() == static_cast<std::size_t>(diag.deviceCount))
        {
            std::cout << "[multi-collective] peer access (1=direct link, 0=disabled):\n";
            for(std::size_t src = 0; src < diag.peerAccess.size(); ++src)
            {
                std::cout << "[multi-collective]    from gpu " << src << ": ";
                for(std::size_t dst = 0; dst < diag.peerAccess[src].size(); ++dst)
                {
                    std::cout << (diag.peerAccess[src][dst] ? '1' : '0');
                    if(dst + 1 < diag.peerAccess[src].size())
                        std::cout << ' ';
                }
                std::cout << '\n';
            }
        }
        else
        {
            std::cout << "[multi-collective] topology: single GPU or peer access data unavailable\n";
        }
    }

    struct CommandLineOptions
    {
        bool verbose = false;
        std::size_t maxDevices = 0; // 0 -> use all available
    };

    CommandLineOptions parseOptions(int argc, char** argv)
    {
        CommandLineOptions opts;
        for(int i = 1; i < argc; ++i)
        {
            std::string arg = argv[i];
            if(arg == "-v" || arg == "--verbose")
            {
                opts.verbose = true;
            }
            else if(arg.rfind("--max-devices=", 0) == 0)
            {
                auto value = arg.substr(std::string{"--max-devices="}.size());
                opts.maxDevices = static_cast<std::size_t>(std::stoul(value));
            }
            else if(arg == "-d" || arg == "--devices")
            {
                if(i + 1 < argc)
                {
                    opts.maxDevices = static_cast<std::size_t>(std::stoul(argv[++i]));
                }
            }
        }
        return opts;
    }

    template<typename Provider, typename Backend>
    int runProviderBackend(Backend const& backend, std::size_t requestedDevices, bool verbose)
    {
        auto exec = backend[alpaka::object::exec];
        auto deviceSpec = backend[alpaka::object::deviceSpec];

        auto selector = alpaka::onHost::makeDeviceSelector(deviceSpec);
        if(!selector.isAvailable())
        {
            if(verbose)
            {
                std::cout << "[multi-collective] no devices available for backend" << std::endl;
            }
            return 0;
        }

        Provider provider;
        if(!provider.isActive())
        {
            if(verbose)
            {
                std::cout << "[multi-collective] provider inactive: " << provider.getBackendName() << '\n';
            }
            return 0;
        }

        if(!provider.supportsMode(at::ICollectiveProvider::ExecutionMode::SingleProcessMultiDevice))
        {
            if(verbose)
            {
                std::cout << "[multi-collective] provider does not support multi-device collectives\n";
            }
            return 0;
        }

        std::size_t available = static_cast<std::size_t>(selector.getDeviceCount());
        std::size_t deviceCount = requestedDevices == 0 ? available : std::min(requestedDevices, available);
        if(deviceCount < 2)
        {
            if(verbose)
            {
                std::cout << "[multi-collective] need at least 2 GPUs (available=" << available << ")\n";
            }
            return 0;
        }

        using DeviceType = decltype(selector.makeDevice(0));
        using QueueType = decltype(std::declval<DeviceType>().makeQueue(alpaka::queueKind::nonBlocking));
        using TensorType = at::Tensor1D<float, DeviceType>;

        struct Participant
        {
            DeviceType device;
            QueueType queue;
            TensorType tensor;
            std::size_t rank;

            Participant(DeviceType dev, QueueType q, TensorType t, std::size_t r)
                : device(std::move(dev)), queue(std::move(q)), tensor(std::move(t)), rank(r)
            {
            }
        };

        constexpr std::size_t elementCount = 32;
        std::vector<Participant> participants;
        participants.reserve(deviceCount);

        for(std::size_t idx = 0; idx < deviceCount; ++idx)
        {
            auto device = selector.makeDevice(static_cast<uint32_t>(idx));
            auto queue = device.makeQueue(alpaka::queueKind::nonBlocking);
            TensorType tensor(device, {elementCount}, "multi_device_tensor");

            auto* hostPtr = tensor.hostData();
            for(std::size_t elem = 0; elem < elementCount; ++elem)
            {
                hostPtr[elem] = static_cast<float>(elem + 1) * static_cast<float>(idx + 1);
            }
            tensor.markHostModified();
            tensor.ensureOnDevice(device, queue);

            participants.emplace_back(std::move(device), std::move(queue), std::move(tensor), idx);
        }

        auto diag = provider.diagnostics();
        if(verbose)
        {
            std::cout << "[multi-collective] backend: " << alpaka::onHost::demangledName(exec) << " / "
                      << alpaka::onHost::demangledName(deviceSpec) << '\n';
            std::cout << "[multi-collective] provider: " << provider.getBackendName() << '\n';
            reportDiagnostics(diag);
        }

        at::ICollectiveProvider::MultiDeviceGroup group;
        group.participants.reserve(participants.size());

        for(auto& participant : participants)
        {
            at::CollectiveExecutionContext ctx{};
            ctx.exec = &exec;
            ctx.device = &participant.device;
            ctx.queue = &participant.queue;
            ctx.globalRank = participant.rank;
            ctx.globalSize = participants.size();
            ctx.localRank = participant.rank;
            ctx.localSize = participants.size();

            if constexpr(requires { alpaka::onHost::getNativeHandle(participant.device); })
            {
                auto nativeDevice = alpaka::onHost::getNativeHandle(participant.device);
                if constexpr(std::is_pointer_v<decltype(nativeDevice)>)
                {
                    ctx.nativeDevice = const_cast<void*>(reinterpret_cast<void const*>(nativeDevice));
                }
                else
                {
                    ctx.nativeDevice = reinterpret_cast<void*>(static_cast<uintptr_t>(nativeDevice));
                }

                if constexpr(std::is_integral_v<decltype(nativeDevice)>)
                {
                    ctx.deviceId = static_cast<int>(nativeDevice);
                }
            }

            if(ctx.deviceId < 0)
            {
                ctx.deviceId = static_cast<int>(participant.rank);
            }

            if constexpr(requires { alpaka::onHost::getNativeHandle(participant.queue); })
            {
                auto nativeQueue = alpaka::onHost::getNativeHandle(participant.queue);
                if constexpr(std::is_pointer_v<decltype(nativeQueue)>)
                {
                    ctx.nativeQueue = const_cast<void*>(reinterpret_cast<void const*>(nativeQueue));
                }
                else
                {
                    ctx.nativeQueue = reinterpret_cast<void*>(static_cast<uintptr_t>(nativeQueue));
                }
            }
            group.participants.emplace_back(ctx);
        }

        auto initStatus = provider.initializeMultiDevice(group);
        if(initStatus != at::OpStatus::Success)
        {
            std::cerr << "[multi-collective] initializeMultiDevice failed for " << provider.getBackendName()
                      << '\n';
            return 1;
        }

        auto dtype = ops::collectiveDataType<float>();

        std::vector<at::ICollectiveProvider::MultiDeviceAllReduceOp> operations;
        operations.reserve(participants.size());

        for(std::size_t idx = 0; idx < participants.size(); ++idx)
        {
            auto& buffer = participants[idx].tensor.deviceBuffer(participants[idx].device, participants[idx].queue);

            at::ICollectiveProvider::MultiDeviceAllReduceOp op{};
            op.localRank = participants[idx].rank;
            op.context = &group.participants[idx];
            op.sendBuffer = buffer.data();
            op.recvBuffer = buffer.data();
            op.elementCount = elementCount;
            op.dtype = dtype;
            op.reduction = ops::CollectiveReduction::Sum;
            op.async = true;
            operations.emplace_back(op);
        }

        auto collectiveStatus = provider.allReduceMultiDevice(operations, /*synchronizeAfter*/ true);
        if(collectiveStatus != at::OpStatus::Success)
        {
            std::cerr << "[multi-collective] allReduceMultiDevice failed with status "
                      << static_cast<int>(collectiveStatus) << '\n';
            return 1;
        }

        for(std::size_t idx = 0; idx < participants.size(); ++idx)
        {
            auto& buffer = participants[idx].tensor.deviceBuffer(participants[idx].device, participants[idx].queue);
            buffer.destructorWaitFor(participants[idx].queue);
        }

        bool validationOk = true;
        float expectedScale = static_cast<float>(participants.size() * (participants.size() + 1)) * 0.5f;

        for(std::size_t idx = 0; idx < participants.size(); ++idx)
        {
            participants[idx].tensor.markDeviceModified(participants[idx].device, participants[idx].queue);
            participants[idx].tensor.toHost(participants[idx].device, participants[idx].queue);
            alpaka::onHost::wait(participants[idx].queue);

            auto* resultPtr = participants[idx].tensor.hostData();
            for(std::size_t elem = 0; elem < elementCount; ++elem)
            {
                float expected = static_cast<float>(elem + 1) * expectedScale;
                if(std::fabs(resultPtr[elem] - expected) > 1e-3f)
                {
                    validationOk = false;
                    if(verbose)
                    {
                        std::cout << "[multi-collective] mismatch on device " << idx << ", element " << elem
                                  << ": expected=" << expected << ", got=" << resultPtr[elem] << '\n';
                    }
                    break;
                }
            }

            if(verbose)
            {
                std::cout << "[multi-collective] device " << idx << " post-reduce values: ";
                for(std::size_t elem = 0; elem < elementCount; ++elem)
                {
                    std::cout << resultPtr[elem];
                    if(elem + 1 < elementCount)
                        std::cout << ' ';
                }
                std::cout << '\n';
            }
        }

        if(!validationOk)
        {
            std::cerr << "[multi-collective] validation failed for " << provider.getBackendName() << '\n';
            return 1;
        }

        std::cout << "[multi-collective] " << provider.getBackendName() << " multi-GPU AllReduce succeeded with "
                  << participants.size() << " participants" << std::endl;
        return 0;
    }

    template<typename Backend>
    int runBackend(Backend const& backend, std::size_t requestedDevices, bool verbose)
    {
        auto exec = backend[alpaka::object::exec];

        using ExecT = std::decay_t<decltype(exec)>;
        if constexpr(std::is_same_v<ExecT, alpaka::exec::GpuHip>)
        {
            return runProviderBackend<at::RCCLProvider>(backend, requestedDevices, verbose);
        }
        else if constexpr(std::is_same_v<ExecT, alpaka::exec::GpuCuda>)
        {
            return runProviderBackend<at::NCCLProvider>(backend, requestedDevices, verbose);
        }
        else
        {
            if(verbose)
            {
                std::cout << "[multi-collective] skipping backend (no RCCL/NCCL support)\n";
            }
            return 0;
        }
    }
} // namespace

int main(int argc, char** argv)
{
    auto options = parseOptions(argc, argv);

    auto runner = [&options](auto const& backend) {
        return runBackend(backend, options.maxDevices, options.verbose);
    };

    return alpaka::onHost::executeForEachIfHasDevice(
        runner,
        alpaka::onHost::allBackends(alpaka::onHost::enabledApis, alpaka::onHost::example::enabledExecutors));
}
