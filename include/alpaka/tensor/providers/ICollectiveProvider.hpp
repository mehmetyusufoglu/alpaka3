/* Collective Provider Interface
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once

#include <alpaka/onHost/interface.hpp>
#include <alpaka/tensor/ops/CollectiveOps.hpp>
#include <alpaka/tensor/providers/ProviderInterface.hpp>

#include <algorithm>
#include <cstddef>
#include <vector>

namespace alpaka::tensor
{
    struct CollectiveExecutionContext
    {
        void const* exec{nullptr};
        void const* device{nullptr};
        void* queue{nullptr};
        void* nativeQueue{nullptr};
        void* nativeDevice{nullptr};
        std::size_t globalRank{0};
        std::size_t globalSize{1};
        std::size_t localRank{0};
        std::size_t localSize{1};
    };

    class ICollectiveProvider : public IOpProvider
    {
    public:
        ~ICollectiveProvider() override = default;

        enum class ExecutionMode
        {
            Undefined,
            SingleProcessSingleDevice,
            SingleProcessMultiDevice,
            MultiProcessSingleDevice,
            MultiProcessMultiDevice
        };

        struct MultiDeviceGroup
        {
            std::vector<CollectiveExecutionContext> participants;
        };

        struct MultiProcessBootstrap
        {
            int worldRank = 0;
            int worldSize = 1;
            std::size_t localRank = 0;
            std::size_t localSize = 1;
            void const* uniqueId = nullptr;
            std::size_t uniqueIdSize = 0;
        };

        virtual bool supportsPattern(ops::CollectivePattern pattern) const = 0;
        virtual bool supportsReduction(ops::CollectiveReduction reduction) const = 0;
        virtual bool supportsDataType(ops::CollectiveDataType dtype) const = 0;

        virtual bool supportsMode(ExecutionMode mode) const
        {
            return mode == ExecutionMode::SingleProcessSingleDevice;
        }

        virtual ExecutionMode currentMode() const = 0;

        virtual OpStatus initializeMultiDevice(MultiDeviceGroup const& group)
        {
            (void)group;
            return OpStatus::Unsupported;
        }

        virtual OpStatus initializeMultiProcess(MultiProcessBootstrap const& bootstrap)
        {
            (void)bootstrap;
            return OpStatus::Unsupported;
        }

        virtual std::size_t localParticipantCount() const
        {
            return 1;
        }

        virtual void setActiveParticipant(std::size_t localRank)
        {
            (void)localRank;
        }

        virtual void synchronizeParticipants()
        {
        }

        template<typename Exec, typename Device, typename Queue>
        OpStatus allReduce(
            Exec const& exec,
            Device const& device,
            Queue& queue,
            void const* sendBuffer,
            void* recvBuffer,
            std::size_t elementCount,
            ops::CollectiveDataType dtype,
            ops::CollectiveReduction reduction,
            bool async)
        {
            CollectiveExecutionContext ctx{
                .exec = static_cast<void const*>(&exec),
                .device = static_cast<void const*>(&device),
                .queue = &queue,
                .nativeQueue = nullptr,
                .globalRank = worldRank(),
                .globalSize = worldSize(),
                .localRank = 0,
                .localSize = localParticipantCount()};
            if constexpr(requires { alpaka::onHost::getNativeHandle(queue); })
            {
                ctx.nativeQueue = reinterpret_cast<void*>(alpaka::onHost::getNativeHandle(queue));
            }
            if constexpr(requires { alpaka::onHost::getNativeHandle(device); })
            {
                ctx.nativeDevice = reinterpret_cast<void*>(alpaka::onHost::getNativeHandle(device));
            }
            if(ctx.localSize > 0)
            {
                std::size_t localRank = worldRank();
                if(localRank >= ctx.localSize)
                    localRank = std::min<std::size_t>(localRank, ctx.localSize - 1);
                ctx.localRank = localRank;
            }
            return allreduce_impl(ctx, sendBuffer, recvBuffer, elementCount, dtype, reduction, async);
        }

        template<typename Exec, typename Device, typename Queue>
        OpStatus broadcast(
            Exec const& exec,
            Device const& device,
            Queue& queue,
            void* buffer,
            std::size_t elementCount,
            ops::CollectiveDataType dtype,
            std::size_t rootRank,
            bool async)
        {
            CollectiveExecutionContext ctx{
                .exec = static_cast<void const*>(&exec),
                .device = static_cast<void const*>(&device),
                .queue = &queue,
                .nativeQueue = nullptr,
                .globalRank = worldRank(),
                .globalSize = worldSize(),
                .localRank = 0,
                .localSize = localParticipantCount()};
            if constexpr(requires { alpaka::onHost::getNativeHandle(queue); })
            {
                ctx.nativeQueue = reinterpret_cast<void*>(alpaka::onHost::getNativeHandle(queue));
            }
            if constexpr(requires { alpaka::onHost::getNativeHandle(device); })
            {
                ctx.nativeDevice = reinterpret_cast<void*>(alpaka::onHost::getNativeHandle(device));
            }
            if(ctx.localSize > 0)
            {
                std::size_t localRank = worldRank();
                if(localRank >= ctx.localSize)
                    localRank = std::min<std::size_t>(localRank, ctx.localSize - 1);
                ctx.localRank = localRank;
            }
            return broadcast_impl(ctx, buffer, elementCount, dtype, rootRank, async);
        }

        template<typename Exec, typename Device, typename Queue>
        OpStatus barrier(Exec const& exec, Device const& device, Queue& queue)
        {
            CollectiveExecutionContext ctx{
                .exec = static_cast<void const*>(&exec),
                .device = static_cast<void const*>(&device),
                .queue = &queue,
                .nativeQueue = nullptr,
                .globalRank = worldRank(),
                .globalSize = worldSize(),
                .localRank = 0,
                .localSize = localParticipantCount()};
            if constexpr(requires { alpaka::onHost::getNativeHandle(queue); })
            {
                ctx.nativeQueue = reinterpret_cast<void*>(alpaka::onHost::getNativeHandle(queue));
            }
            if constexpr(requires { alpaka::onHost::getNativeHandle(device); })
            {
                ctx.nativeDevice = reinterpret_cast<void*>(alpaka::onHost::getNativeHandle(device));
            }
            if(ctx.localSize > 0)
            {
                std::size_t localRank = worldRank();
                if(localRank >= ctx.localSize)
                    localRank = std::min<std::size_t>(localRank, ctx.localSize - 1);
                ctx.localRank = localRank;
            }
            return barrier_impl(ctx);
        }

        virtual std::size_t worldSize() const = 0;
        virtual std::size_t worldRank() const = 0;

    protected:
        virtual OpStatus allreduce_impl(
            CollectiveExecutionContext const&,
            void const*,
            void*,
            std::size_t,
            ops::CollectiveDataType,
            ops::CollectiveReduction,
            bool)
            = 0;

        virtual OpStatus broadcast_impl(
            CollectiveExecutionContext const&,
            void*,
            std::size_t,
            ops::CollectiveDataType,
            std::size_t,
            bool)
            = 0;

        virtual OpStatus barrier_impl(CollectiveExecutionContext const&) = 0;
    };
} // namespace alpaka::tensor
