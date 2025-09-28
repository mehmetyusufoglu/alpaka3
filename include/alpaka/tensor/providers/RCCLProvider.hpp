/* RCCL Collective Provider
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once

#include <alpaka/tensor/ops/CollectiveOps.hpp>
#include <alpaka/tensor/providers/ICollectiveProvider.hpp>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(ALPAKA_HAS_RCCL) && defined(ALPAKA_ACC_GPU_HIP_ENABLED)
#    include <hip/hip_runtime_api.h>
#    include <rccl/rccl.h>
#endif

namespace alpaka::tensor
{
    class RCCLProvider : public ICollectiveProvider
    {
    public:
        using ExecutionMode = ICollectiveProvider::ExecutionMode;
        using MultiDeviceGroup = ICollectiveProvider::MultiDeviceGroup;
        using MultiProcessBootstrap = ICollectiveProvider::MultiProcessBootstrap;

        struct Diagnostics
        {
            int deviceCount = 0;
            int activeDevice = -1;
            std::string activeDeviceName;
            std::vector<std::string> deviceNames;
            std::vector<std::vector<bool>> peerAccess;
        };

        RCCLProvider() = default;
        ~RCCLProvider() override;

        std::string getBackendName() const override;
        bool supportsOperation(OpType op) const override;
        bool isActive() const override;
        bool supportsMode(ExecutionMode mode) const override;
        ExecutionMode currentMode() const override;

        Diagnostics diagnostics() const;

        bool supportsPattern(ops::CollectivePattern pattern) const override;
        bool supportsReduction(ops::CollectiveReduction reduction) const override;
        bool supportsDataType(ops::CollectiveDataType dtype) const override;

        std::size_t worldSize() const override;
        std::size_t worldRank() const override;

        OpStatus initializeMultiDevice(MultiDeviceGroup const& group) override;
        OpStatus initializeMultiProcess(MultiProcessBootstrap const& bootstrap) override;
        std::size_t localParticipantCount() const override;
        void setActiveParticipant(std::size_t localRank) override;
        void synchronizeParticipants() override;

    protected:
        OpStatus allreduce_impl(
            CollectiveExecutionContext const& ctx,
            void const* sendBuffer,
            void* recvBuffer,
            std::size_t elementCount,
            ops::CollectiveDataType dtype,
            ops::CollectiveReduction reduction,
            bool async) override;

        OpStatus broadcast_impl(
            CollectiveExecutionContext const& ctx,
            void* buffer,
            std::size_t elementCount,
            ops::CollectiveDataType dtype,
            std::size_t rootRank,
            bool async) override;

        OpStatus barrier_impl(CollectiveExecutionContext const& ctx) override;

    private:
        void ensureInitialized() const;
        void finalize() noexcept;
        void finalizeMultiDevice() const;
        void finalizeSingleDevice() const;
        void finalizeMultiProcess() const;
    OpStatus initializeSingleDevice() const;
    std::size_t resolveLocalRank(CollectiveExecutionContext const& ctx) const;
#if defined(ALPAKA_HAS_RCCL) && defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    ncclComm_t resolveCommunicator(std::size_t localRank) const;
#endif
        void updateWorldTracking(std::size_t localRank) const;
        void setDeviceForRank(std::size_t localRank) const;

#if defined(ALPAKA_HAS_RCCL) && defined(ALPAKA_ACC_GPU_HIP_ENABLED)
        ncclDataType_t mapDataType(ops::CollectiveDataType dtype) const;
        ncclRedOp_t mapReduction(ops::CollectiveReduction reduction) const;
#endif

    private:
#if defined(ALPAKA_HAS_RCCL) && defined(ALPAKA_ACC_GPU_HIP_ENABLED)
        mutable RCCLProvider::ExecutionMode mode_ = RCCLProvider::ExecutionMode::Undefined;
        mutable ncclComm_t singleDeviceComm_ = nullptr;
        mutable bool singleDeviceInitialized_ = false;
        mutable std::vector<ncclComm_t> multiDeviceComms_{};
        mutable std::vector<int> multiDeviceDeviceIds_{};
        mutable std::vector<void*> multiDeviceQueues_{};
        mutable std::size_t activeParticipant_ = 0;
        mutable ncclComm_t multiProcessComm_ = nullptr;
        mutable bool active_ = false;
        mutable int worldSize_ = 1;
        mutable int worldRank_ = 0;
        mutable std::size_t localSize_ = 1;
#else
        static constexpr bool active_ = false;
#endif
    };
} // namespace alpaka::tensor

#if defined(ALPAKA_HAS_RCCL) && defined(ALPAKA_ACC_GPU_HIP_ENABLED)
namespace alpaka::tensor
{
    inline RCCLProvider::~RCCLProvider()
    {
        finalize();
    }

    inline std::string RCCLProvider::getBackendName() const
    {
        return "RCCL (HIP Collective)";
    }

    inline bool RCCLProvider::supportsOperation(OpType op) const
    {
        return op == OpType::Collective;
    }

    inline bool RCCLProvider::isActive() const
    {
        ensureInitialized();
        return active_;
    }

    inline bool RCCLProvider::supportsPattern(ops::CollectivePattern pattern) const
    {
        return pattern == ops::CollectivePattern::AllReduce || pattern == ops::CollectivePattern::Broadcast
               || pattern == ops::CollectivePattern::Barrier;
    }

    inline bool RCCLProvider::supportsReduction(ops::CollectiveReduction reduction) const
    {
        switch(reduction)
        {
        case ops::CollectiveReduction::Sum:
        case ops::CollectiveReduction::Product:
        case ops::CollectiveReduction::Maximum:
        case ops::CollectiveReduction::Minimum:
            return true;
        default:
            return false;
        }
    }

    inline bool RCCLProvider::supportsDataType(ops::CollectiveDataType dtype) const
    {
        switch(dtype)
        {
        case ops::CollectiveDataType::Float16:
        case ops::CollectiveDataType::BFloat16:
        case ops::CollectiveDataType::Float32:
        case ops::CollectiveDataType::Float64:
        case ops::CollectiveDataType::Int8:
        case ops::CollectiveDataType::UInt8:
        case ops::CollectiveDataType::Int32:
        case ops::CollectiveDataType::UInt32:
        case ops::CollectiveDataType::Int64:
            return true;
        default:
            return false;
        }
    }

    inline std::size_t RCCLProvider::worldSize() const
    {
        ensureInitialized();
        return static_cast<std::size_t>(worldSize_);
    }

    inline std::size_t RCCLProvider::worldRank() const
    {
        ensureInitialized();
        return static_cast<std::size_t>(worldRank_);
    }

    inline RCCLProvider::Diagnostics RCCLProvider::diagnostics() const
    {
        Diagnostics diag;

        int deviceCount = 0;
        hipError_t hipStatus = hipGetDeviceCount(&deviceCount);
        if(hipStatus != hipSuccess || deviceCount <= 0)
        {
            return diag;
        }

        diag.deviceCount = deviceCount;

        int activeDevice = -1;
        hipStatus = hipGetDevice(&activeDevice);
        if(hipStatus == hipSuccess)
        {
            diag.activeDevice = activeDevice;
        }

        diag.deviceNames.resize(static_cast<std::size_t>(deviceCount));
        diag.peerAccess.assign(static_cast<std::size_t>(deviceCount), std::vector<bool>(static_cast<std::size_t>(deviceCount), false));

        for(int dev = 0; dev < deviceCount; ++dev)
        {
            hipDeviceProp_t props{};
            hipStatus = hipGetDeviceProperties(&props, dev);
            if(hipStatus == hipSuccess)
            {
                diag.deviceNames[static_cast<std::size_t>(dev)] = props.name;
                if(dev == diag.activeDevice)
                {
                    diag.activeDeviceName = props.name;
                }
            }
            else
            {
                diag.deviceNames[static_cast<std::size_t>(dev)] = "unknown";
            }
        }

        for(int src = 0; src < deviceCount; ++src)
        {
            for(int dst = 0; dst < deviceCount; ++dst)
            {
                if(src == dst)
                {
                    diag.peerAccess[static_cast<std::size_t>(src)][static_cast<std::size_t>(dst)] = true;
                    continue;
                }

                int canAccess = 0;
                hipStatus = hipDeviceCanAccessPeer(&canAccess, src, dst);
                if(hipStatus == hipSuccess)
                {
                    diag.peerAccess[static_cast<std::size_t>(src)][static_cast<std::size_t>(dst)] = (canAccess != 0);
                }
            }
        }

        return diag;
    }

    inline ncclDataType_t RCCLProvider::mapDataType(ops::CollectiveDataType dtype) const
    {
        switch(dtype)
        {
        case ops::CollectiveDataType::Float16:
            return ncclFloat16;
        case ops::CollectiveDataType::BFloat16:
            return ncclBfloat16;
        case ops::CollectiveDataType::Float32:
            return ncclFloat;
        case ops::CollectiveDataType::Float64:
            return ncclDouble;
        case ops::CollectiveDataType::Int8:
            return ncclInt8;
        case ops::CollectiveDataType::UInt8:
            return ncclUint8;
        case ops::CollectiveDataType::Int32:
            return ncclInt;
        case ops::CollectiveDataType::UInt32:
            return ncclUint32;
        case ops::CollectiveDataType::Int64:
            return ncclInt64;
        default:
            throw std::invalid_argument("RCCL does not support requested collective datatype");
        }
    }

    inline ncclRedOp_t RCCLProvider::mapReduction(ops::CollectiveReduction reduction) const
    {
        switch(reduction)
        {
        case ops::CollectiveReduction::Sum:
            return ncclSum;
        case ops::CollectiveReduction::Product:
            return ncclProd;
        case ops::CollectiveReduction::Maximum:
            return ncclMax;
        case ops::CollectiveReduction::Minimum:
            return ncclMin;
        default:
            throw std::invalid_argument("RCCL reduction not supported");
        }
    }

    inline void RCCLProvider::ensureInitialized() const
    {
    if(mode_ != RCCLProvider::ExecutionMode::Undefined || singleDeviceInitialized_)
            return;

        initializeSingleDevice();
    }

    inline void RCCLProvider::finalize() noexcept
    {
        finalizeSingleDevice();
        finalizeMultiDevice();
        finalizeMultiProcess();

    mode_ = RCCLProvider::ExecutionMode::Undefined;
        active_ = false;
        worldSize_ = 1;
        worldRank_ = 0;
        localSize_ = 1;
        activeParticipant_ = 0;
    }

    inline void RCCLProvider::finalizeSingleDevice() const
    {
        if(singleDeviceComm_ != nullptr)
        {
            ncclCommDestroy(singleDeviceComm_);
            singleDeviceComm_ = nullptr;
        }
        singleDeviceInitialized_ = false;
    }

    inline void RCCLProvider::finalizeMultiDevice() const
    {
        if(!multiDeviceComms_.empty())
        {
            for(ncclComm_t comm : multiDeviceComms_)
            {
                if(comm != nullptr)
                {
                    ncclCommDestroy(comm);
                }
            }
        }
        multiDeviceComms_.clear();
        multiDeviceDeviceIds_.clear();
        multiDeviceQueues_.clear();
    }

    inline void RCCLProvider::finalizeMultiProcess() const
    {
        if(multiProcessComm_ != nullptr)
        {
            ncclCommDestroy(multiProcessComm_);
            multiProcessComm_ = nullptr;
        }
    }

    inline OpStatus RCCLProvider::initializeSingleDevice() const
    {
        if(singleDeviceInitialized_ && singleDeviceComm_ != nullptr)
        {
            mode_ = RCCLProvider::ExecutionMode::SingleProcessSingleDevice;
            active_ = true;
            worldSize_ = 1;
            worldRank_ = 0;
            localSize_ = 1;
            return OpStatus::Success;
        }

        int deviceId = 0;
        hipError_t hipStatus = hipGetDevice(&deviceId);
        if(hipStatus != hipSuccess)
        {
            singleDeviceInitialized_ = true;
            active_ = false;
            return OpStatus::Unsupported;
        }

        ncclResult_t status = ncclCommInitAll(&singleDeviceComm_, 1, &deviceId);
        if(status != ncclSuccess)
        {
            singleDeviceComm_ = nullptr;
            singleDeviceInitialized_ = true;
            active_ = false;
            return OpStatus::Error;
        }

    singleDeviceInitialized_ = true;
    mode_ = RCCLProvider::ExecutionMode::SingleProcessSingleDevice;
        active_ = true;
        worldSize_ = 1;
        worldRank_ = 0;
        localSize_ = 1;
        activeParticipant_ = 0;
        return OpStatus::Success;
    }

    inline bool RCCLProvider::supportsMode(RCCLProvider::ExecutionMode mode) const
    {
        switch(mode)
        {
        case RCCLProvider::ExecutionMode::Undefined:
        case RCCLProvider::ExecutionMode::SingleProcessSingleDevice:
        case RCCLProvider::ExecutionMode::SingleProcessMultiDevice:
        case RCCLProvider::ExecutionMode::MultiProcessSingleDevice:
            return true;
        default:
            return false;
        }
    }

    inline RCCLProvider::ExecutionMode RCCLProvider::currentMode() const
    {
        return mode_;
    }

    inline OpStatus RCCLProvider::initializeMultiDevice(MultiDeviceGroup const& group)
    {
        if(group.participants.empty())
            return OpStatus::Unsupported;

        finalize();

        std::size_t count = group.participants.size();
        multiDeviceComms_.resize(count, nullptr);
        multiDeviceDeviceIds_.resize(count);
        multiDeviceQueues_.resize(count, nullptr);

        for(std::size_t i = 0; i < count; ++i)
        {
            auto const& participant = group.participants[i];
            multiDeviceDeviceIds_[i] = static_cast<int>(participant.localRank);
            multiDeviceQueues_[i] = participant.queue;
        }

        ncclResult_t status = ncclCommInitAll(
            multiDeviceComms_.data(),
            static_cast<int>(count),
            multiDeviceDeviceIds_.data());
        if(status != ncclSuccess)
        {
            finalize();
            return OpStatus::Error;
        }

    mode_ = RCCLProvider::ExecutionMode::SingleProcessMultiDevice;
        active_ = true;
        worldSize_ = static_cast<int>(count);
        worldRank_ = 0;
        localSize_ = count;
        activeParticipant_ = 0;
        setDeviceForRank(activeParticipant_);
        return OpStatus::Success;
    }

    inline OpStatus RCCLProvider::initializeMultiProcess(MultiProcessBootstrap const& bootstrap)
    {
        if(bootstrap.uniqueId == nullptr || bootstrap.uniqueIdSize != sizeof(ncclUniqueId))
        {
            return OpStatus::Unsupported;
        }

        finalize();

        auto const* id = static_cast<ncclUniqueId const*>(bootstrap.uniqueId);
        ncclComm_t comm = nullptr;
        ncclResult_t status = ncclCommInitRank(
            &comm,
            static_cast<int>(bootstrap.worldSize),
            *id,
            static_cast<int>(bootstrap.worldRank));
        if(status != ncclSuccess)
        {
            finalize();
            return OpStatus::Error;
        }

        multiProcessComm_ = comm;
    mode_ = RCCLProvider::ExecutionMode::MultiProcessSingleDevice;
        active_ = true;
        worldSize_ = static_cast<int>(bootstrap.worldSize);
        worldRank_ = static_cast<int>(bootstrap.worldRank);
        localSize_ = bootstrap.localSize;
        activeParticipant_ = bootstrap.localRank;
        return OpStatus::Success;
    }

    inline std::size_t RCCLProvider::localParticipantCount() const
    {
        return localSize_;
    }

    inline void RCCLProvider::setActiveParticipant(std::size_t localRank)
    {
        activeParticipant_ = localRank;
        updateWorldTracking(localRank);
        setDeviceForRank(localRank);
    }

    inline void RCCLProvider::synchronizeParticipants()
    {
        switch(mode_)
        {
        case RCCLProvider::ExecutionMode::SingleProcessMultiDevice:
        {
            for(std::size_t i = 0; i < multiDeviceDeviceIds_.size(); ++i)
            {
                setDeviceForRank(i);
                hipDeviceSynchronize();
            }
            break;
        }
        case RCCLProvider::ExecutionMode::SingleProcessSingleDevice:
        case RCCLProvider::ExecutionMode::MultiProcessSingleDevice:
        default:
            hipDeviceSynchronize();
            break;
        }
    }

    inline std::size_t RCCLProvider::resolveLocalRank(CollectiveExecutionContext const& ctx) const
    {
        if(mode_ == RCCLProvider::ExecutionMode::SingleProcessMultiDevice && ctx.queue != nullptr)
        {
            for(std::size_t i = 0; i < multiDeviceQueues_.size(); ++i)
            {
                if(multiDeviceQueues_[i] == ctx.queue)
                    return i;
            }
        }

        if(ctx.localRank < multiDeviceComms_.size())
            return ctx.localRank;

        return activeParticipant_;
    }

    inline ncclComm_t RCCLProvider::resolveCommunicator(std::size_t localRank) const
    {
        switch(mode_)
        {
        case RCCLProvider::ExecutionMode::SingleProcessSingleDevice:
            return singleDeviceComm_;
        case RCCLProvider::ExecutionMode::SingleProcessMultiDevice:
            return (localRank < multiDeviceComms_.size()) ? multiDeviceComms_[localRank] : nullptr;
        case RCCLProvider::ExecutionMode::MultiProcessSingleDevice:
            return multiProcessComm_;
        default:
            return nullptr;
        }
    }

    inline void RCCLProvider::updateWorldTracking(std::size_t localRank) const
    {
        switch(mode_)
        {
        case RCCLProvider::ExecutionMode::SingleProcessMultiDevice:
            worldSize_ = static_cast<int>(multiDeviceComms_.size());
            worldRank_ = static_cast<int>(localRank);
            break;
        case RCCLProvider::ExecutionMode::SingleProcessSingleDevice:
            worldSize_ = 1;
            worldRank_ = 0;
            break;
        case RCCLProvider::ExecutionMode::MultiProcessSingleDevice:
            break;
        default:
            break;
        }
    }

    inline void RCCLProvider::setDeviceForRank(std::size_t localRank) const
    {
        if(mode_ != RCCLProvider::ExecutionMode::SingleProcessMultiDevice)
            return;

        if(localRank >= multiDeviceDeviceIds_.size())
            return;

        int deviceId = multiDeviceDeviceIds_[localRank];
        hipSetDevice(deviceId);
    }

    inline OpStatus RCCLProvider::allreduce_impl(
        CollectiveExecutionContext const& ctx,
        void const* sendBuffer,
        void* recvBuffer,
        std::size_t elementCount,
        ops::CollectiveDataType dtype,
        ops::CollectiveReduction reduction,
        bool async)
    {
        ensureInitialized();
        if(!active_)
            return OpStatus::Unsupported;

        std::size_t localRank = resolveLocalRank(ctx);
        setDeviceForRank(localRank);

        ncclComm_t communicator = resolveCommunicator(localRank);
        if(communicator == nullptr)
            return OpStatus::Unsupported;

        hipStream_t hipStream = ctx.nativeQueue ? static_cast<hipStream_t>(ctx.nativeQueue) : nullptr;
        ncclResult_t status = ncclAllReduce(
            sendBuffer,
            recvBuffer,
            elementCount,
            mapDataType(dtype),
            mapReduction(reduction),
            communicator,
            hipStream);
        if(status != ncclSuccess)
            return OpStatus::Error;

        if(!async)
        {
            if(hipStream != nullptr)
            {
                hipError_t hipStatus = hipStreamSynchronize(hipStream);
                if(hipStatus != hipSuccess)
                    return OpStatus::Error;
            }
            else
            {
                hipError_t hipStatus = hipDeviceSynchronize();
                if(hipStatus != hipSuccess)
                    return OpStatus::Error;
            }
        }
        return OpStatus::Success;
    }

    inline OpStatus RCCLProvider::broadcast_impl(
        CollectiveExecutionContext const& ctx,
        void* buffer,
        std::size_t elementCount,
        ops::CollectiveDataType dtype,
        std::size_t rootRank,
        bool async)
    {
        ensureInitialized();
        if(!active_)
            return OpStatus::Unsupported;

        std::size_t localRank = resolveLocalRank(ctx);
        setDeviceForRank(localRank);

        ncclComm_t communicator = resolveCommunicator(localRank);
        if(communicator == nullptr)
            return OpStatus::Unsupported;

        hipStream_t hipStream = ctx.nativeQueue ? static_cast<hipStream_t>(ctx.nativeQueue) : nullptr;
        ncclResult_t status = ncclBroadcast(
            static_cast<void const*>(buffer),
            buffer,
            elementCount,
            mapDataType(dtype),
            static_cast<int>(rootRank),
            communicator,
            hipStream);
        if(status != ncclSuccess)
            return OpStatus::Error;

        if(!async)
        {
            if(hipStream != nullptr)
            {
                hipError_t hipStatus = hipStreamSynchronize(hipStream);
                if(hipStatus != hipSuccess)
                    return OpStatus::Error;
            }
            else
            {
                hipError_t hipStatus = hipDeviceSynchronize();
                if(hipStatus != hipSuccess)
                    return OpStatus::Error;
            }
        }
        return OpStatus::Success;
    }

    inline OpStatus RCCLProvider::barrier_impl(CollectiveExecutionContext const& ctx)
    {
        ensureInitialized();
        if(!active_)
            return OpStatus::Unsupported;

        std::size_t localRank = resolveLocalRank(ctx);
        setDeviceForRank(localRank);

        ncclComm_t communicator = resolveCommunicator(localRank);
        if(communicator == nullptr)
            return OpStatus::Unsupported;

        hipStream_t hipStream = ctx.nativeQueue ? static_cast<hipStream_t>(ctx.nativeQueue) : nullptr;
        if(hipStream != nullptr)
        {
            hipError_t hipStatus = hipStreamSynchronize(hipStream);
            if(hipStatus != hipSuccess)
                return OpStatus::Error;
        }
        else
        {
            hipError_t hipStatus = hipDeviceSynchronize();
            if(hipStatus != hipSuccess)
                return OpStatus::Error;
        }
        return OpStatus::Success;
    }
} // namespace alpaka::tensor
#else
namespace alpaka::tensor
{
    inline RCCLProvider::~RCCLProvider() = default;

    inline std::string RCCLProvider::getBackendName() const
    {
        return "RCCL (unavailable)";
    }

    inline bool RCCLProvider::supportsOperation(OpType) const
    {
        return false;
    }

    inline bool RCCLProvider::isActive() const
    {
        return false;
    }

    inline bool RCCLProvider::supportsMode(RCCLProvider::ExecutionMode) const
    {
        return false;
    }

    inline RCCLProvider::ExecutionMode RCCLProvider::currentMode() const
    {
        return RCCLProvider::ExecutionMode::Undefined;
    }

    inline bool RCCLProvider::supportsPattern(ops::CollectivePattern) const
    {
        return false;
    }

    inline bool RCCLProvider::supportsReduction(ops::CollectiveReduction) const
    {
        return false;
    }

    inline bool RCCLProvider::supportsDataType(ops::CollectiveDataType) const
    {
        return false;
    }

    inline std::size_t RCCLProvider::worldSize() const
    {
        return 1;
    }

    inline std::size_t RCCLProvider::worldRank() const
    {
        return 0;
    }

    inline OpStatus RCCLProvider::initializeMultiDevice(MultiDeviceGroup const&)
    {
        return OpStatus::Unsupported;
    }

    inline OpStatus RCCLProvider::initializeMultiProcess(MultiProcessBootstrap const&)
    {
        return OpStatus::Unsupported;
    }

    inline std::size_t RCCLProvider::localParticipantCount() const
    {
        return 1;
    }

    inline void RCCLProvider::setActiveParticipant(std::size_t)
    {
    }

    inline void RCCLProvider::synchronizeParticipants()
    {
    }

    inline RCCLProvider::Diagnostics RCCLProvider::diagnostics() const
    {
        return Diagnostics{};
    }

    inline void RCCLProvider::ensureInitialized() const
    {
    }

    inline void RCCLProvider::finalize() noexcept
    {
    }

    inline OpStatus RCCLProvider::allreduce_impl(
        CollectiveExecutionContext const&,
        void const*,
        void*,
        std::size_t,
        ops::CollectiveDataType,
        ops::CollectiveReduction,
        bool)
    {
        return OpStatus::Unsupported;
    }

    inline OpStatus RCCLProvider::broadcast_impl(
        CollectiveExecutionContext const&,
        void*,
        std::size_t,
        ops::CollectiveDataType,
        std::size_t,
        bool)
    {
        return OpStatus::Unsupported;
    }

    inline OpStatus RCCLProvider::barrier_impl(CollectiveExecutionContext const&)
    {
        return OpStatus::Unsupported;
    }
} // namespace alpaka::tensor
#endif
