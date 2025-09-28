/* NCCL Collective Provider
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once

#include <alpaka/tensor/ops/CollectiveOps.hpp>
#include <alpaka/tensor/providers/ICollectiveProvider.hpp>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef ALPAKA_TENSOR_USE_NCCL
#    if defined(ALPAKA_HAS_NCCL)
#        if defined(__has_include)
#            if __has_include(<cuda_runtime_api.h>) && __has_include(<nccl.h>)
#                define ALPAKA_TENSOR_USE_NCCL 1
#            else
#                define ALPAKA_TENSOR_USE_NCCL 0
#            endif
#        else
#            define ALPAKA_TENSOR_USE_NCCL 1
#        endif
#    else
#        define ALPAKA_TENSOR_USE_NCCL 0
#    endif
#endif

#if ALPAKA_TENSOR_USE_NCCL
#    include <cuda_runtime_api.h>
#    include <nccl.h>
#endif

namespace alpaka::tensor
{
    class NCCLProvider : public ICollectiveProvider
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

        NCCLProvider() = default;
        ~NCCLProvider() override;

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
#if ALPAKA_TENSOR_USE_NCCL
    ncclComm_t resolveCommunicator(std::size_t localRank) const;
#endif
    void updateWorldTracking(std::size_t localRank) const;
    void setCudaDeviceForRank(std::size_t localRank) const;

#if ALPAKA_TENSOR_USE_NCCL
        ncclDataType_t mapDataType(ops::CollectiveDataType dtype) const;
        ncclRedOp_t mapReduction(ops::CollectiveReduction reduction) const;
#endif

    private:
#if ALPAKA_TENSOR_USE_NCCL
        mutable NCCLProvider::ExecutionMode mode_ = NCCLProvider::ExecutionMode::Undefined;
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

#if ALPAKA_TENSOR_USE_NCCL
namespace alpaka::tensor
{
    inline NCCLProvider::~NCCLProvider()
    {
        finalize();
    }

    inline std::string NCCLProvider::getBackendName() const
    {
        return "NCCL (CUDA Collective)";
    }

    inline bool NCCLProvider::supportsOperation(OpType op) const
    {
        return op == OpType::Collective;
    }

    inline bool NCCLProvider::isActive() const
    {
        ensureInitialized();
        return active_;
    }

    inline bool NCCLProvider::supportsPattern(ops::CollectivePattern pattern) const
    {
        return pattern == ops::CollectivePattern::AllReduce || pattern == ops::CollectivePattern::Broadcast
               || pattern == ops::CollectivePattern::Barrier;
    }

    inline bool NCCLProvider::supportsReduction(ops::CollectiveReduction reduction) const
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

    inline bool NCCLProvider::supportsDataType(ops::CollectiveDataType dtype) const
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

    inline std::size_t NCCLProvider::worldSize() const
    {
        ensureInitialized();
        return static_cast<std::size_t>(worldSize_);
    }

    inline std::size_t NCCLProvider::worldRank() const
    {
        ensureInitialized();
        return static_cast<std::size_t>(worldRank_);
    }

    inline NCCLProvider::Diagnostics NCCLProvider::diagnostics() const
    {
        Diagnostics diag;

        int deviceCount = 0;
        cudaError_t cudaStatus = cudaGetDeviceCount(&deviceCount);
        if(cudaStatus != cudaSuccess || deviceCount <= 0)
        {
            return diag;
        }

        diag.deviceCount = deviceCount;

        int activeDevice = -1;
        cudaStatus = cudaGetDevice(&activeDevice);
        if(cudaStatus == cudaSuccess)
        {
            diag.activeDevice = activeDevice;
        }

        diag.deviceNames.resize(static_cast<std::size_t>(deviceCount));
        diag.peerAccess.assign(static_cast<std::size_t>(deviceCount), std::vector<bool>(static_cast<std::size_t>(deviceCount), false));

        for(int dev = 0; dev < deviceCount; ++dev)
        {
            cudaDeviceProp props{};
            cudaStatus = cudaGetDeviceProperties(&props, dev);
            if(cudaStatus == cudaSuccess)
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
                cudaStatus = cudaDeviceCanAccessPeer(&canAccess, src, dst);
                if(cudaStatus == cudaSuccess)
                {
                    diag.peerAccess[static_cast<std::size_t>(src)][static_cast<std::size_t>(dst)] = (canAccess != 0);
                }
            }
        }

        return diag;
    }

    inline ncclDataType_t NCCLProvider::mapDataType(ops::CollectiveDataType dtype) const
    {
        switch(dtype)
        {
        case ops::CollectiveDataType::Float16:
            return ncclHalf;
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
            throw std::invalid_argument("NCCL does not support requested collective datatype");
        }
    }

    inline ncclRedOp_t NCCLProvider::mapReduction(ops::CollectiveReduction reduction) const
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
            throw std::invalid_argument("NCCL reduction not supported");
        }
    }

    inline void NCCLProvider::ensureInitialized() const
    {
    if(mode_ != NCCLProvider::ExecutionMode::Undefined || singleDeviceInitialized_)
            return;

        initializeSingleDevice();
    }

    inline void NCCLProvider::finalize() noexcept
    {
        finalizeSingleDevice();
        finalizeMultiDevice();
        finalizeMultiProcess();

    mode_ = NCCLProvider::ExecutionMode::Undefined;
        active_ = false;
        worldSize_ = 1;
        worldRank_ = 0;
        localSize_ = 1;
        activeParticipant_ = 0;
    }

    inline void NCCLProvider::finalizeSingleDevice() const
    {
        if(singleDeviceComm_ != nullptr)
        {
            ncclCommDestroy(singleDeviceComm_);
            singleDeviceComm_ = nullptr;
        }
        singleDeviceInitialized_ = false;
    }

    inline void NCCLProvider::finalizeMultiDevice() const
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

    inline void NCCLProvider::finalizeMultiProcess() const
    {
        if(multiProcessComm_ != nullptr)
        {
            ncclCommDestroy(multiProcessComm_);
            multiProcessComm_ = nullptr;
        }
    }

    inline OpStatus NCCLProvider::initializeSingleDevice() const
    {
        if(singleDeviceInitialized_ && singleDeviceComm_ != nullptr)
        {
            mode_ = NCCLProvider::ExecutionMode::SingleProcessSingleDevice;
            active_ = true;
            worldSize_ = 1;
            worldRank_ = 0;
            localSize_ = 1;
            return OpStatus::Success;
        }

        int deviceId = 0;
        cudaError_t cudaStatus = cudaGetDevice(&deviceId);
        if(cudaStatus != cudaSuccess)
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
    mode_ = NCCLProvider::ExecutionMode::SingleProcessSingleDevice;
        active_ = true;
        worldSize_ = 1;
        worldRank_ = 0;
        localSize_ = 1;
        activeParticipant_ = 0;
        return OpStatus::Success;
    }

    inline bool NCCLProvider::supportsMode(NCCLProvider::ExecutionMode mode) const
    {
        switch(mode)
        {
        case NCCLProvider::ExecutionMode::Undefined:
        case NCCLProvider::ExecutionMode::SingleProcessSingleDevice:
        case NCCLProvider::ExecutionMode::SingleProcessMultiDevice:
        case NCCLProvider::ExecutionMode::MultiProcessSingleDevice:
            return true;
        default:
            return false;
        }
    }

    inline NCCLProvider::ExecutionMode NCCLProvider::currentMode() const
    {
        return mode_;
    }

    inline OpStatus NCCLProvider::initializeMultiDevice(MultiDeviceGroup const& group)
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

    mode_ = NCCLProvider::ExecutionMode::SingleProcessMultiDevice;
        active_ = true;
        worldSize_ = static_cast<int>(count);
        worldRank_ = 0;
        localSize_ = count;
        activeParticipant_ = 0;
        setCudaDeviceForRank(activeParticipant_);
        return OpStatus::Success;
    }

    inline OpStatus NCCLProvider::initializeMultiProcess(MultiProcessBootstrap const& bootstrap)
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
    mode_ = NCCLProvider::ExecutionMode::MultiProcessSingleDevice;
        active_ = true;
        worldSize_ = static_cast<int>(bootstrap.worldSize);
        worldRank_ = static_cast<int>(bootstrap.worldRank);
        localSize_ = bootstrap.localSize;
        activeParticipant_ = bootstrap.localRank;
        return OpStatus::Success;
    }

    inline std::size_t NCCLProvider::localParticipantCount() const
    {
        return localSize_;
    }

    inline void NCCLProvider::setActiveParticipant(std::size_t localRank)
    {
        activeParticipant_ = localRank;
        updateWorldTracking(localRank);
        setCudaDeviceForRank(localRank);
    }

    inline void NCCLProvider::synchronizeParticipants()
    {
        switch(mode_)
        {
    case NCCLProvider::ExecutionMode::SingleProcessMultiDevice:
        {
            for(std::size_t i = 0; i < multiDeviceDeviceIds_.size(); ++i)
            {
                setCudaDeviceForRank(i);
                cudaDeviceSynchronize();
            }
            break;
        }
    case NCCLProvider::ExecutionMode::SingleProcessSingleDevice:
    case NCCLProvider::ExecutionMode::MultiProcessSingleDevice:
        default:
            cudaDeviceSynchronize();
            break;
        }
    }

    inline std::size_t NCCLProvider::resolveLocalRank(CollectiveExecutionContext const& ctx) const
    {
    if(mode_ == NCCLProvider::ExecutionMode::SingleProcessMultiDevice && ctx.queue != nullptr)
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

    inline ncclComm_t NCCLProvider::resolveCommunicator(std::size_t localRank) const
    {
        switch(mode_)
        {
    case NCCLProvider::ExecutionMode::SingleProcessSingleDevice:
            return singleDeviceComm_;
    case NCCLProvider::ExecutionMode::SingleProcessMultiDevice:
            return (localRank < multiDeviceComms_.size()) ? multiDeviceComms_[localRank] : nullptr;
    case NCCLProvider::ExecutionMode::MultiProcessSingleDevice:
            return multiProcessComm_;
        default:
            return nullptr;
        }
    }

    inline void NCCLProvider::updateWorldTracking(std::size_t localRank) const
    {
        switch(mode_)
        {
    case NCCLProvider::ExecutionMode::SingleProcessMultiDevice:
            worldSize_ = static_cast<int>(multiDeviceComms_.size());
            worldRank_ = static_cast<int>(localRank);
            break;
    case NCCLProvider::ExecutionMode::SingleProcessSingleDevice:
            worldSize_ = 1;
            worldRank_ = 0;
            break;
    case NCCLProvider::ExecutionMode::MultiProcessSingleDevice:
            break;
        default:
            break;
        }
    }

    inline void NCCLProvider::setCudaDeviceForRank(std::size_t localRank) const
    {
        if(mode_ != ExecutionMode::SingleProcessMultiDevice)
            return;

        if(localRank >= multiDeviceDeviceIds_.size())
            return;

        int deviceId = multiDeviceDeviceIds_[localRank];
        cudaSetDevice(deviceId);
    }

    inline OpStatus NCCLProvider::allreduce_impl(
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
        setCudaDeviceForRank(localRank);

        ncclComm_t communicator = resolveCommunicator(localRank);
        if(communicator == nullptr)
            return OpStatus::Unsupported;

        cudaStream_t cudaStream = ctx.nativeQueue ? static_cast<cudaStream_t>(ctx.nativeQueue) : nullptr;
        ncclResult_t status = ncclAllReduce(
            sendBuffer,
            recvBuffer,
            elementCount,
            mapDataType(dtype),
            mapReduction(reduction),
            communicator,
            cudaStream);
        if(status != ncclSuccess)
            return OpStatus::Error;

        if(!async)
        {
            if(cudaStream != nullptr)
            {
                cudaError_t cudaStatus = cudaStreamSynchronize(cudaStream);
                if(cudaStatus != cudaSuccess)
                    return OpStatus::Error;
            }
            else
            {
                cudaError_t cudaStatus = cudaDeviceSynchronize();
                if(cudaStatus != cudaSuccess)
                    return OpStatus::Error;
            }
        }
        return OpStatus::Success;
    }

    inline OpStatus NCCLProvider::broadcast_impl(
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
        setCudaDeviceForRank(localRank);

        ncclComm_t communicator = resolveCommunicator(localRank);
        if(communicator == nullptr)
            return OpStatus::Unsupported;

        cudaStream_t cudaStream = ctx.nativeQueue ? static_cast<cudaStream_t>(ctx.nativeQueue) : nullptr;
        ncclResult_t status = ncclBroadcast(
            static_cast<void const*>(buffer),
            buffer,
            elementCount,
            mapDataType(dtype),
            static_cast<int>(rootRank),
            communicator,
            cudaStream);
        if(status != ncclSuccess)
            return OpStatus::Error;

        if(!async)
        {
            if(cudaStream != nullptr)
            {
                cudaError_t cudaStatus = cudaStreamSynchronize(cudaStream);
                if(cudaStatus != cudaSuccess)
                    return OpStatus::Error;
            }
            else
            {
                cudaError_t cudaStatus = cudaDeviceSynchronize();
                if(cudaStatus != cudaSuccess)
                    return OpStatus::Error;
            }
        }
        return OpStatus::Success;
    }

    inline OpStatus NCCLProvider::barrier_impl(CollectiveExecutionContext const& ctx)
    {
        ensureInitialized();
        if(!active_)
            return OpStatus::Unsupported;

        std::size_t localRank = resolveLocalRank(ctx);
        setCudaDeviceForRank(localRank);

        ncclComm_t communicator = resolveCommunicator(localRank);
        if(communicator == nullptr)
            return OpStatus::Unsupported;

        cudaStream_t cudaStream = ctx.nativeQueue ? static_cast<cudaStream_t>(ctx.nativeQueue) : nullptr;
        if(cudaStream != nullptr)
        {
            cudaError_t cudaStatus = cudaStreamSynchronize(cudaStream);
            if(cudaStatus != cudaSuccess)
                return OpStatus::Error;
        }
        else
        {
            cudaError_t cudaStatus = cudaDeviceSynchronize();
            if(cudaStatus != cudaSuccess)
                return OpStatus::Error;
        }
        return OpStatus::Success;
    }
} // namespace alpaka::tensor
#else
namespace alpaka::tensor
{
    inline NCCLProvider::~NCCLProvider() = default;

    inline std::string NCCLProvider::getBackendName() const
    {
        return "NCCL (unavailable)";
    }

    inline bool NCCLProvider::supportsOperation(OpType) const
    {
        return false;
    }

    inline bool NCCLProvider::isActive() const
    {
        return false;
    }

    inline bool NCCLProvider::supportsMode(NCCLProvider::ExecutionMode) const
    {
        return false;
    }

    inline NCCLProvider::ExecutionMode NCCLProvider::currentMode() const
    {
        return NCCLProvider::ExecutionMode::Undefined;
    }

    inline bool NCCLProvider::supportsPattern(ops::CollectivePattern) const
    {
        return false;
    }

    inline bool NCCLProvider::supportsReduction(ops::CollectiveReduction) const
    {
        return false;
    }

    inline bool NCCLProvider::supportsDataType(ops::CollectiveDataType) const
    {
        return false;
    }

    inline std::size_t NCCLProvider::worldSize() const
    {
        return 1;
    }

    inline std::size_t NCCLProvider::worldRank() const
    {
        return 0;
    }

    inline OpStatus NCCLProvider::initializeMultiDevice(MultiDeviceGroup const&)
    {
        return OpStatus::Unsupported;
    }

    inline OpStatus NCCLProvider::initializeMultiProcess(MultiProcessBootstrap const&)
    {
        return OpStatus::Unsupported;
    }

    inline std::size_t NCCLProvider::localParticipantCount() const
    {
        return 1;
    }

    inline void NCCLProvider::setActiveParticipant(std::size_t)
    {
    }

    inline void NCCLProvider::synchronizeParticipants()
    {
    }

    inline NCCLProvider::Diagnostics NCCLProvider::diagnostics() const
    {
        return Diagnostics{};
    }

    inline void NCCLProvider::ensureInitialized() const
    {
    }

    inline void NCCLProvider::finalize() noexcept
    {
    }

    inline OpStatus NCCLProvider::allreduce_impl(
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

    inline OpStatus NCCLProvider::broadcast_impl(
        CollectiveExecutionContext const&,
        void*,
        std::size_t,
        ops::CollectiveDataType,
        std::size_t,
        bool)
    {
        return OpStatus::Unsupported;
    }

    inline OpStatus NCCLProvider::barrier_impl(CollectiveExecutionContext const&)
    {
        return OpStatus::Unsupported;
    }
} // namespace alpaka::tensor
#endif
