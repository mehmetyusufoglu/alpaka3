/* Collective Provider Interface
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once

#include <alpaka/onHost/interface.hpp>
#include <alpaka/tensor/ops/CollectiveOps.hpp>
#include <alpaka/tensor/providers/ProviderInterface.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <type_traits>
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
        int deviceId{-1};
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

        /**
         * @brief Structure describing a single all-reduce operation for multi-device execution.
         * 
         * This struct contains all the information needed to perform an all-reduce operation
         * on a specific device/rank within a multi-device collective. Multiple operations
         * of this type can be batched together for concurrent execution.
         */
        struct MultiDeviceAllReduceOp
        {
            std::size_t localRank{0};                             ///< Local rank/device index within the collective
            CollectiveExecutionContext const* context{nullptr};    ///< Execution context (device, queue, etc.)
            void const* sendBuffer{nullptr};                       ///< Input data buffer
            void* recvBuffer{nullptr};                             ///< Output data buffer (can be same as sendBuffer)
            std::size_t elementCount{0};                           ///< Number of elements in the buffer
            ops::CollectiveDataType dtype{ops::CollectiveDataType::Float32};  ///< Data type of elements
            ops::CollectiveReduction reduction{ops::CollectiveReduction::Sum}; ///< Reduction operation
            bool async{false};                                     ///< Whether to synchronize immediately after the operation
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

        /**
         * @brief Perform batched all-reduce operations across multiple devices.
         *
         * This method allows efficient execution of multiple all-reduce operations
         * concurrently, which is essential for multi-GPU collective communication
         * where all participants must call the operation simultaneously.
         *
         * For providers like NCCL, this method uses either group semantics
         * (ncclGroupStart/ncclGroupEnd) or launches operations concurrently
         * using threads to avoid deadlocks.
         *
         * @param operations Vector of all-reduce operations to execute
         * @param synchronizeAfter If true, wait for all operations to complete before returning
         * @return OpStatus indicating success, error, or unsupported
         */
        virtual OpStatus allReduceMultiDevice(
            std::vector<MultiDeviceAllReduceOp> const& operations,
            bool synchronizeAfter)
        {
            (void)operations;
            (void)synchronizeAfter;
            return OpStatus::Unsupported;
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
                auto nativeDevice = alpaka::onHost::getNativeHandle(device);
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
                auto nativeDevice = alpaka::onHost::getNativeHandle(device);
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
                auto nativeDevice = alpaka::onHost::getNativeHandle(device);
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
