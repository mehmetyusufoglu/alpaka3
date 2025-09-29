/* NCCL Collective Provider
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once

#include <alpaka/tensor/ops/CollectiveOps.hpp>
#include <alpaka/tensor/providers/ICollectiveProvider.hpp>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
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
        
        /**
         * @brief Perform batched all-reduce operations across multiple NCCL communicators.
         *
         * This implementation handles multi-device all-reduce operations efficiently by:
         * 1. First attempting to use NCCL group semantics (ncclGroupStart/ncclGroupEnd)
         *    to batch operations for optimal performance
         * 2. Falling back to threaded execution if group operations fail
         * 3. Properly handling device context switching and stream management
         * 4. Providing comprehensive error reporting and status aggregation
         *
         * The method ensures that all NCCL operations are launched concurrently,
         * which is required to avoid deadlocks in collective communication.
         *
         * @param operations Vector of all-reduce operations to execute concurrently
         * @param synchronizeAfter Whether to synchronize all participants after completion
         * @return OpStatus::Success if all operations succeeded, Error if any failed, 
         *         Unsupported if provider is not in multi-device mode
         */
        OpStatus allReduceMultiDevice(
            std::vector<ICollectiveProvider::MultiDeviceAllReduceOp> const& operations,
            bool synchronizeAfter) override;
        
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
        int resolveDeviceId(CollectiveExecutionContext const& participant, std::size_t fallbackIndex) const;
        static void reportCudaError(char const* expr, cudaError_t status, char const* file, int line);
        static void reportNcclError(char const* expr, ncclResult_t status, char const* file, int line);
        
        /**
         * @brief Enqueue an NCCL all-reduce operation for a specific rank.
         *
         * This helper method handles device context setup, stream selection,
         * and NCCL communicator resolution for a single all-reduce operation.
         * It's designed to be called either within NCCL groups or from worker threads.
         *
         * @param localRank The rank/device index for this operation
         * @param ctx Execution context containing device and queue information
         * @param sendBuffer Input data buffer
         * @param recvBuffer Output data buffer
         * @param elementCount Number of elements to reduce
         * @param dtype Data type of the elements
         * @param reduction Type of reduction operation
         * @param outStream Returns the CUDA stream used for the operation
         * @return ncclResult_t indicating the NCCL operation status
         */
        ncclResult_t enqueueAllReduce(
            std::size_t localRank,
            CollectiveExecutionContext const& ctx,
            void const* sendBuffer,
            void* recvBuffer,
            std::size_t elementCount,
            ops::CollectiveDataType dtype,
            ops::CollectiveReduction reduction,
            cudaStream_t& outStream) const;
            
        /**
         * @brief Synchronize a CUDA stream for a specific rank.
         *
         * This method ensures proper device context and synchronizes the given stream.
         * It's used to wait for NCCL operations to complete.
         *
         * @param localRank The rank/device index
         * @param stream The CUDA stream to synchronize
         * @return OpStatus indicating success or failure
         */
        OpStatus synchronizeRankStream(std::size_t localRank, cudaStream_t stream) const;
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
    mutable std::vector<cudaStream_t> multiDeviceStreams_{};
        mutable std::size_t activeParticipant_ = 0;
        mutable ncclComm_t multiProcessComm_ = nullptr;
        mutable bool active_ = false;
        mutable int worldSize_ = 1;
        mutable int worldRank_ = 0;
        mutable std::size_t localSize_ = 1;
        mutable bool groupFallbackThreadsEnabled_ = false;
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

    inline ncclResult_t NCCLProvider::enqueueAllReduce(
        std::size_t localRank,
        CollectiveExecutionContext const& ctx,
        void const* sendBuffer,
        void* recvBuffer,
        std::size_t elementCount,
        ops::CollectiveDataType dtype,
        ops::CollectiveReduction reduction,
        cudaStream_t& outStream) const
    {
        if(localRank < multiDeviceQueues_.size())
        {
            multiDeviceQueues_[localRank] = ctx.queue;
        }

        setCudaDeviceForRank(localRank);

        ncclComm_t communicator = resolveCommunicator(localRank);
        if(communicator == nullptr)
        {
            outStream = nullptr;
            return ncclSystemError;
        }

        cudaStream_t cudaStream = nullptr;
        if(ctx.nativeQueue != nullptr)
        {
            cudaStream = static_cast<cudaStream_t>(ctx.nativeQueue);
        }
        else if(mode_ == NCCLProvider::ExecutionMode::SingleProcessMultiDevice && localRank < multiDeviceStreams_.size())
        {
            cudaStream = multiDeviceStreams_[localRank];
        }

        outStream = cudaStream;

        return ncclAllReduce(
            sendBuffer,
            recvBuffer,
            elementCount,
            mapDataType(dtype),
            mapReduction(reduction),
            communicator,
            cudaStream);
    }

    inline OpStatus NCCLProvider::synchronizeRankStream(std::size_t localRank, cudaStream_t stream) const
    {
        setCudaDeviceForRank(localRank);

        cudaError_t const cudaStatus = (stream != nullptr) ? cudaStreamSynchronize(stream) : cudaDeviceSynchronize();
        if(cudaStatus != cudaSuccess)
        {
            reportCudaError(
                stream != nullptr ? "cudaStreamSynchronize" : "cudaDeviceSynchronize",
                cudaStatus,
                __FILE__,
                __LINE__);
            return OpStatus::Error;
        }
        return OpStatus::Success;
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
        if(multiDeviceComms_.empty() && multiDeviceStreams_.empty())
        {
            return;
        }

        if(!multiDeviceComms_.empty())
        {
            std::cout << "[NCCL-DEBUG] finalizeMultiDevice: destroying NCCL communicators" << std::endl;
            for(std::size_t rank = 0; rank < multiDeviceComms_.size(); ++rank)
            {
                ncclComm_t comm = multiDeviceComms_[rank];
                if(comm == nullptr)
                    continue;

    #if defined(NCCL_VERSION_CODE) && (NCCL_VERSION_CODE >= 2700)
                ncclResult_t finalizeStatus = ncclCommFinalize(comm);
                if(finalizeStatus == ncclInProgress)
                {
                    ncclResult_t const abortStatus = ncclCommAbort(comm);
                    if(abortStatus != ncclSuccess)
                    {
                        reportNcclError("ncclCommAbort", abortStatus, __FILE__, __LINE__);
                    }
                    reportNcclError("ncclCommFinalize", finalizeStatus, __FILE__, __LINE__);
                }
                else if(finalizeStatus != ncclSuccess)
                {
                    reportNcclError("ncclCommFinalize", finalizeStatus, __FILE__, __LINE__);
                }
    #endif

                ncclResult_t const destroyStatus = ncclCommDestroy(comm);
                if(destroyStatus != ncclSuccess)
                {
                    reportNcclError("ncclCommDestroy", destroyStatus, __FILE__, __LINE__);
                }
            }
        }

        if(!multiDeviceStreams_.empty())
        {
            std::cout << "[NCCL-DEBUG] finalizeMultiDevice: destroying CUDA streams" << std::endl;
            for(std::size_t rank = 0; rank < multiDeviceStreams_.size(); ++rank)
            {
                cudaStream_t stream = multiDeviceStreams_[rank];
                if(stream == nullptr)
                    continue;

                int deviceId = (rank < multiDeviceDeviceIds_.size()) ? multiDeviceDeviceIds_[rank] : -1;
                if(deviceId >= 0)
                {
                    cudaError_t const setStatus = cudaSetDevice(deviceId);
                    if(setStatus != cudaSuccess)
                    {
                        reportCudaError("cudaSetDevice", setStatus, __FILE__, __LINE__);
                    }
                }

                cudaError_t const destroyStatus = cudaStreamDestroy(stream);
                if(destroyStatus != cudaSuccess)
                {
                    reportCudaError("cudaStreamDestroy", destroyStatus, __FILE__, __LINE__);
                }
            }
        }

        multiDeviceComms_.clear();
        multiDeviceDeviceIds_.clear();
        multiDeviceQueues_.clear();
        multiDeviceStreams_.clear();
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
            reportCudaError("cudaGetDevice", cudaStatus, __FILE__, __LINE__);
            singleDeviceInitialized_ = true;
            active_ = false;
            return OpStatus::Unsupported;
        }

        ncclResult_t status = ncclCommInitAll(&singleDeviceComm_, 1, &deviceId);
        if(status != ncclSuccess)
        {
            reportNcclError("ncclCommInitAll", status, __FILE__, __LINE__);
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
        std::cout << "[NCCL-DEBUG] initializeMultiDevice called with " 
                  << group.participants.size() << " participants" << std::endl;
        
        if(group.participants.empty())
        {
            std::cout << "[NCCL-DEBUG] No participants, returning Unsupported" << std::endl;
            return OpStatus::Unsupported;
        }

        std::cout << "[NCCL-DEBUG] Calling finalize()..." << std::endl;
        finalize();
        std::cout << "[NCCL-DEBUG] finalize() completed" << std::endl;

        std::size_t const count = group.participants.size();
        std::cout << "[NCCL-DEBUG] Setting up data structures for " << count << " devices" << std::endl;

        multiDeviceComms_.assign(count, nullptr);
        multiDeviceDeviceIds_.assign(count, -1);
        multiDeviceQueues_.assign(count, nullptr);
        multiDeviceStreams_.assign(count, nullptr);
        groupFallbackThreadsEnabled_ = true;

        if(char const* enableGroups = std::getenv("ALPAKA_NCCL_ENABLE_GROUPS"))
        {
            if(std::atoi(enableGroups) != 0)
            {
                std::cout << "[NCCL-DEBUG] Enabling NCCL group path via ALPAKA_NCCL_ENABLE_GROUPS" << std::endl;
                groupFallbackThreadsEnabled_ = false;
            }
            else
            {
                std::cout << "[NCCL-DEBUG] Disabling NCCL group path via ALPAKA_NCCL_ENABLE_GROUPS=0" << std::endl;
            }
        }

        if(char const* forceFallback = std::getenv("ALPAKA_NCCL_FORCE_FALLBACK"))
        {
            if(forceFallback[0] != '\0' && std::atoi(forceFallback) != 0)
            {
                std::cout << "[NCCL-DEBUG] Forcing thread fallback path via ALPAKA_NCCL_FORCE_FALLBACK" << std::endl;
                groupFallbackThreadsEnabled_ = true;
            }
        }

        std::cout << "[NCCL-DEBUG] Step 1: Extracting device IDs and setting up CUDA contexts..." << std::endl;
        for(std::size_t i = 0; i < count; ++i)
        {
            auto const& participant = group.participants[i];
            int const deviceId = resolveDeviceId(participant, i);
            std::cout << "[NCCL-DEBUG]   Participant " << i << ": deviceId=" << deviceId 
                      << ", localRank=" << participant.localRank << std::endl;
            
            if(deviceId < 0)
            {
                std::cout << "[NCCL-DEBUG] Invalid device ID, returning Unsupported" << std::endl;
                finalize();
                return OpStatus::Unsupported;
            }

            multiDeviceDeviceIds_[i] = deviceId;
            multiDeviceQueues_[i] = participant.queue;

            cudaError_t const setStatus = cudaSetDevice(deviceId);
            if(setStatus != cudaSuccess)
            {
                reportCudaError("cudaSetDevice", setStatus, __FILE__, __LINE__);
                finalize();
                return OpStatus::Error;
            }

            cudaError_t const warmupStatus = cudaFree(nullptr);
            if(warmupStatus != cudaSuccess && warmupStatus != cudaErrorInvalidValue)
            {
                reportCudaError("cudaFree(nullptr)", warmupStatus, __FILE__, __LINE__);
                finalize();
                return OpStatus::Error;
            }
        }

        for(std::size_t rank = 0; rank < count; ++rank)
        {
            if(multiDeviceDeviceIds_[rank] < 0)
            {
                std::cout << "[NCCL-DEBUG] Missing device assignment for local rank " << rank << std::endl;
                finalize();
                return OpStatus::Unsupported;
            }
        }

        std::cout << "[NCCL-DEBUG] Step 2: Acquiring NCCL unique identifier" << std::endl;
        ncclUniqueId uniqueId{};
        ncclResult_t const getIdStatus = ncclGetUniqueId(&uniqueId);
        if(getIdStatus != ncclSuccess)
        {
            std::cout << "[NCCL-DEBUG] ncclGetUniqueId failed: " << ncclGetErrorString(getIdStatus) << std::endl;
            reportNcclError("ncclGetUniqueId", getIdStatus, __FILE__, __LINE__);
            finalize();
            return OpStatus::Error;
        }

        std::cout << "[NCCL-DEBUG] Step 3: Creating NCCL communicators with group semantics" << std::endl;
        ncclResult_t const groupStartStatus = ncclGroupStart();
        if(groupStartStatus != ncclSuccess)
        {
            std::cout << "[NCCL-DEBUG] ncclGroupStart failed during communicator init: "
                      << ncclGetErrorString(groupStartStatus) << std::endl;
            reportNcclError("ncclGroupStart", groupStartStatus, __FILE__, __LINE__);
            finalize();
            return OpStatus::Error;
        }

        bool initError = false;
        for(std::size_t i = 0; i < count; ++i)
        {
            cudaError_t const setStatus = cudaSetDevice(multiDeviceDeviceIds_[i]);
            if(setStatus != cudaSuccess)
            {
                reportCudaError("cudaSetDevice", setStatus, __FILE__, __LINE__);
                initError = true;
                continue;
            }

            ncclResult_t const initStatus = ncclCommInitRank(
                &multiDeviceComms_[i],
                static_cast<int>(count),
                uniqueId,
                static_cast<int>(i));
            if(initStatus != ncclSuccess)
            {
                std::cout << "[NCCL-DEBUG] ncclCommInitRank failed for rank " << i << ": "
                          << ncclGetErrorString(initStatus) << std::endl;
                reportNcclError("ncclCommInitRank", initStatus, __FILE__, __LINE__);
                initError = true;
            }
            else
            {
                std::cout << "[NCCL-DEBUG]   Queued communicator init for rank " << i << " (device "
                          << multiDeviceDeviceIds_[i] << ")" << std::endl;
            }
        }

        ncclResult_t const groupEndStatus = ncclGroupEnd();
        if(groupEndStatus != ncclSuccess)
        {
            std::cout << "[NCCL-DEBUG] ncclGroupEnd failed during communicator init: "
                      << ncclGetErrorString(groupEndStatus) << std::endl;
            reportNcclError("ncclGroupEnd", groupEndStatus, __FILE__, __LINE__);
            initError = true;
        }

        if(initError)
        {
            finalize();
            return OpStatus::Error;
        }

    std::cout << "[NCCL-DEBUG] Step 4: Using participant-provided streams for NCCL operations" << std::endl;

        mode_ = NCCLProvider::ExecutionMode::SingleProcessMultiDevice;
        active_ = true;
        worldSize_ = static_cast<int>(count);
        worldRank_ = 0;
        localSize_ = count;
        activeParticipant_ = 0;
        setCudaDeviceForRank(activeParticipant_);
        
        std::cout << "[NCCL-DEBUG] Multi-device initialization completed successfully!" << std::endl;
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
            reportNcclError("ncclCommInitRank", status, __FILE__, __LINE__);
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
                cudaError_t const status = cudaDeviceSynchronize();
                if(status != cudaSuccess)
                    reportCudaError("cudaDeviceSynchronize", status, __FILE__, __LINE__);
            }
            break;
        }
    case NCCLProvider::ExecutionMode::SingleProcessSingleDevice:
    case NCCLProvider::ExecutionMode::MultiProcessSingleDevice:
        default:
        {
                cudaError_t const status = cudaDeviceSynchronize();
                if(status != cudaSuccess)
                    reportCudaError("cudaDeviceSynchronize", status, __FILE__, __LINE__);
            break;
        }
        }
    }

    inline OpStatus NCCLProvider::allReduceMultiDevice(
        std::vector<ICollectiveProvider::MultiDeviceAllReduceOp> const& operations,
        bool synchronizeAfter)
    {
        ensureInitialized();
        if(operations.empty())
        {
            if(synchronizeAfter)
            {
                synchronizeParticipants();
            }
            return OpStatus::Success;
        }

        if(!active_)
        {
            return OpStatus::Unsupported;
        }

        bool const isMultiDevice = mode_ == NCCLProvider::ExecutionMode::SingleProcessMultiDevice;
        OpStatus aggregateStatus = OpStatus::Success;

        // Fallback path for providers that are not in multi-device mode.
        if(!multiDeviceComms_.empty())
        {
            for(std::size_t idx = 0; idx < multiDeviceComms_.size(); ++idx)
            {
                ncclComm_t comm = multiDeviceComms_[idx];
                if(comm == nullptr)
                    continue;

                int deviceId = (idx < multiDeviceDeviceIds_.size()) ? multiDeviceDeviceIds_[idx] : -1;
                if(deviceId >= 0)
                {
                    cudaError_t const setStatus = cudaSetDevice(deviceId);
                    if(setStatus != cudaSuccess)
                    {
                        reportCudaError("cudaSetDevice", setStatus, __FILE__, __LINE__);
                    }
                }

                cudaError_t const syncStatus = cudaDeviceSynchronize();
                if(syncStatus != cudaSuccess)
                {
                    reportCudaError("cudaDeviceSynchronize", syncStatus, __FILE__, __LINE__);
                }

                ncclResult_t const abortStatus = ncclCommAbort(comm);
                if(abortStatus != ncclSuccess && abortStatus != ncclInvalidUsage)
                {
                    reportNcclError("ncclCommAbort", abortStatus, __FILE__, __LINE__);
                }

                ncclResult_t const destroyStatus = ncclCommDestroy(comm);
                if(destroyStatus != ncclSuccess)
                {
                    reportNcclError("ncclCommDestroy", destroyStatus, __FILE__, __LINE__);
                }
            }
        }
                return false;
            }
            return true;
        };

        bool invalidOperationDetected = false;
        for(std::size_t idx = 0; idx < operations.size(); ++idx)
        {
            if(!validateOperation(idx))
            {
                invalidOperationDetected = true;
            }
        }

        if(invalidOperationDetected)
        {
            std::cout << "[NCCL-DEBUG] allReduceMultiDevice aborting: invalid operation detected" << std::endl;
            bool hasUnsupported = std::any_of(
                opStatuses.begin(),
                opStatuses.end(),
                [](OpStatus status) { return status == OpStatus::Unsupported; });
            return hasUnsupported ? OpStatus::Unsupported : OpStatus::Error;
        }

        bool usedGroup = false;
        bool groupFailed = false;

    if(!groupFallbackThreadsEnabled_)
        {
            ncclResult_t const groupStart = ncclGroupStart();
            if(groupStart == ncclSuccess)
            {
        std::cout << "[NCCL-DEBUG] allReduceMultiDevice using NCCL group path" << std::endl;
                usedGroup = true;

                for(std::size_t idx = 0; idx < operations.size(); ++idx)
                {
                    auto const& op = operations[idx];
                    ncclResult_t const enqueueStatus = enqueueAllReduce(
                        op.localRank,
                        *op.context,
                        op.sendBuffer,
                        op.recvBuffer,
                        op.elementCount,
                        op.dtype,
                        op.reduction,
                        streams[idx]);

                    ncclStatuses[idx] = enqueueStatus;
                    std::cout << "[NCCL-DEBUG] Enqueued rank " << op.localRank << " on stream "
                              << static_cast<void*>(streams[idx]) << std::endl;
                    std::cout << "[NCCL-DEBUG]   Rank " << op.localRank
                              << " enqueue returned " << ncclGetErrorString(enqueueStatus) << std::endl;
                    if(enqueueStatus != ncclSuccess)
                    {
                        std::cout << "[NCCL-DEBUG] Group enqueue failed for rank " << op.localRank
                                  << ": " << ncclGetErrorString(enqueueStatus) << std::endl;
                        opStatuses[idx] = OpStatus::Error;
                        groupFailed = true;
                    }
                }

                if(!operations.empty())
                {
                    setCudaDeviceForRank(operations.front().localRank);
                }
                std::cout << "[NCCL-DEBUG] Finalizing NCCL group" << std::endl;
                ncclResult_t const groupEnd = ncclGroupEnd();
                std::cout << "[NCCL-DEBUG] ncclGroupEnd returned " << ncclGetErrorString(groupEnd)
                          << std::endl;
                if(groupEnd != ncclSuccess)
                {
                    std::cout << "[NCCL-DEBUG] ncclGroupEnd returned error: " << ncclGetErrorString(groupEnd)
                              << std::endl;
                    groupFailed = true;
                    if(operations.size() > 0)
                    {
                        ncclStatuses[0] = groupEnd;
                        opStatuses[0] = OpStatus::Error;
                    }
                }
            }
            else
            {
                std::cout << "[NCCL-DEBUG] ncclGroupStart failed: " << ncclGetErrorString(groupStart) << std::endl;
                reportNcclError("ncclGroupStart", groupStart, __FILE__, __LINE__);
                groupFailed = true;
                groupFallbackThreadsEnabled_ = true;
            }
        }

        if(usedGroup && !groupFailed)
        {
            groupFallbackThreadsEnabled_ = false;

            for(std::size_t idx = 0; idx < operations.size(); ++idx)
            {
                if(ncclStatuses[idx] != ncclSuccess)
                {
                    opStatuses[idx] = OpStatus::Error;
                    continue;
                }

                auto const& op = operations[idx];
                if(!op.async || synchronizeAfter)
                {
                    opStatuses[idx] = synchronizeRankStream(op.localRank, streams[idx]);
                }
                else
                {
                    opStatuses[idx] = OpStatus::Success;
                }
            }
        }

        if(!usedGroup || groupFailed)
        {
            if(usedGroup && groupFailed)
            {
                groupFallbackThreadsEnabled_ = true;
            }

            std::cout << "[NCCL-DEBUG] allReduceMultiDevice using thread fallback path" << std::endl;

            std::fill(streams.begin(), streams.end(), nullptr);
            std::fill(ncclStatuses.begin(), ncclStatuses.end(), ncclSuccess);
            std::fill(opStatuses.begin(), opStatuses.end(), OpStatus::Success);

            std::vector<std::thread> workers;
            workers.reserve(operations.size());

            for(std::size_t idx = 0; idx < operations.size(); ++idx)
            {
                workers.emplace_back([&, idx]() {
                    if(!validateOperation(idx))
                    {
                        return;
                    }

                    auto const& op = operations[idx];
                    cudaStream_t stream = nullptr;
                    ncclResult_t const enqueueStatus = enqueueAllReduce(
                        op.localRank,
                        *op.context,
                        op.sendBuffer,
                        op.recvBuffer,
                        op.elementCount,
                        op.dtype,
                        op.reduction,
                        stream);

                    streams[idx] = stream;
                    ncclStatuses[idx] = enqueueStatus;

                    if(enqueueStatus != ncclSuccess)
                    {
                        std::cout << "[NCCL-DEBUG] Thread enqueue failed for rank " << op.localRank
                                  << ": " << ncclGetErrorString(enqueueStatus) << std::endl;
                        opStatuses[idx] = OpStatus::Error;
                        return;
                    }

                    if(!op.async || synchronizeAfter)
                    {
                        opStatuses[idx] = synchronizeRankStream(op.localRank, stream);
                        if(opStatuses[idx] != OpStatus::Success)
                        {
                            std::cout << "[NCCL-DEBUG] Thread sync failed for rank " << op.localRank << std::endl;
                        }
                    }
                    else
                    {
                        opStatuses[idx] = OpStatus::Success;
                    }
                });
            }

            for(auto& worker : workers)
            {
                if(worker.joinable())
                {
                    worker.join();
                }
            }
        }

        bool hasUnsupported = false;
        bool hasError = false;

        for(std::size_t idx = 0; idx < operations.size(); ++idx)
        {
            if(ncclStatuses[idx] != ncclSuccess)
            {
                reportNcclError("ncclAllReduce", ncclStatuses[idx], __FILE__, __LINE__);
                hasError = true;
            }

            switch(opStatuses[idx])
            {
            case OpStatus::Error:
                hasError = true;
                break;
            case OpStatus::Unsupported:
                hasUnsupported = true;
                break;
            default:
                break;
            }
        }

        if(!hasError && synchronizeAfter)
        {
            synchronizeParticipants();
        }

        if(hasError)
        {
            return OpStatus::Error;
        }
        if(hasUnsupported)
        {
            return OpStatus::Unsupported;
        }
        return OpStatus::Success;
    }

    inline std::size_t NCCLProvider::resolveLocalRank(CollectiveExecutionContext const& ctx) const
    {
        if(mode_ == NCCLProvider::ExecutionMode::SingleProcessMultiDevice)
        {
            if(ctx.deviceId >= 0)
            {
                for(std::size_t i = 0; i < multiDeviceDeviceIds_.size(); ++i)
                {
                    if(multiDeviceDeviceIds_[i] == ctx.deviceId)
                        return i;
                }
            }

            if(ctx.queue != nullptr)
            {
                for(std::size_t i = 0; i < multiDeviceQueues_.size(); ++i)
                {
                    if(multiDeviceQueues_[i] == ctx.queue)
                        return i;
                }
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

    inline int NCCLProvider::resolveDeviceId(CollectiveExecutionContext const& participant, std::size_t fallbackIndex) const
    {
        if(participant.deviceId >= 0)
            return participant.deviceId;

        if(participant.nativeDevice != nullptr)
            return static_cast<int>(reinterpret_cast<uintptr_t>(participant.nativeDevice));

        return static_cast<int>(fallbackIndex);
    }

    inline void NCCLProvider::reportCudaError(char const* expr, cudaError_t status, char const* file, int line)
    {
        if(status == cudaSuccess)
            return;

        std::cerr << "[alpaka][tensor][NCCL] CUDA call '" << expr << "' failed with '" << cudaGetErrorString(status)
                  << "' (code " << static_cast<int>(status) << ") at " << file << ':' << line << std::endl;
    }

    inline void NCCLProvider::reportNcclError(char const* expr, ncclResult_t status, char const* file, int line)
    {
        if(status == ncclSuccess)
            return;

        std::cerr << "[alpaka][tensor][NCCL] NCCL call '" << expr << "' failed with '" << ncclGetErrorString(status)
                  << "' (code " << static_cast<int>(status) << ") at " << file << ':' << line << std::endl;
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

        std::size_t const localRank = resolveLocalRank(ctx);
        cudaStream_t stream = nullptr;
        ncclResult_t const enqueueStatus = enqueueAllReduce(
            localRank,
            ctx,
            sendBuffer,
            recvBuffer,
            elementCount,
            dtype,
            reduction,
            stream);

        if(enqueueStatus != ncclSuccess)
        {
            reportNcclError("ncclAllReduce", enqueueStatus, __FILE__, __LINE__);
            return OpStatus::Error;
        }

        if(!async)
        {
            OpStatus const syncStatus = synchronizeRankStream(localRank, stream);
            if(syncStatus != OpStatus::Success)
            {
                return syncStatus;
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

        if(localRank < multiDeviceQueues_.size())
            multiDeviceQueues_[localRank] = ctx.queue;

        cudaStream_t cudaStream = nullptr;
        if(mode_ == NCCLProvider::ExecutionMode::SingleProcessMultiDevice && localRank < multiDeviceStreams_.size())
        {
            cudaStream = multiDeviceStreams_[localRank];
        }
        if(cudaStream == nullptr && ctx.nativeQueue != nullptr)
        {
            cudaStream = static_cast<cudaStream_t>(ctx.nativeQueue);
        }
        ncclResult_t status = ncclBroadcast(
            static_cast<void const*>(buffer),
            buffer,
            elementCount,
            mapDataType(dtype),
            static_cast<int>(rootRank),
            communicator,
            cudaStream);
        if(status != ncclSuccess)
        {
            reportNcclError("ncclBroadcast", status, __FILE__, __LINE__);
            return OpStatus::Error;
        }

        if(!async)
        {
            cudaError_t const cudaStatus = (cudaStream != nullptr) ? cudaStreamSynchronize(cudaStream) : cudaDeviceSynchronize();
            if(cudaStatus != cudaSuccess)
            {
                reportCudaError(cudaStream != nullptr ? "cudaStreamSynchronize" : "cudaDeviceSynchronize", cudaStatus, __FILE__, __LINE__);
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

        if(localRank < multiDeviceQueues_.size())
            multiDeviceQueues_[localRank] = ctx.queue;

        cudaStream_t cudaStream = nullptr;
        if(mode_ == NCCLProvider::ExecutionMode::SingleProcessMultiDevice && localRank < multiDeviceStreams_.size())
        {
            cudaStream = multiDeviceStreams_[localRank];
        }
        if(cudaStream == nullptr && ctx.nativeQueue != nullptr)
        {
            cudaStream = static_cast<cudaStream_t>(ctx.nativeQueue);
        }

        cudaError_t const cudaStatus = (cudaStream != nullptr) ? cudaStreamSynchronize(cudaStream) : cudaDeviceSynchronize();
        if(cudaStatus != cudaSuccess)
        {
            reportCudaError(cudaStream != nullptr ? "cudaStreamSynchronize" : "cudaDeviceSynchronize", cudaStatus, __FILE__, __LINE__);
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

    inline OpStatus NCCLProvider::allReduceMultiDevice(
        std::vector<ICollectiveProvider::MultiDeviceAllReduceOp> const& operations,
        bool synchronizeAfter)
    {
        (void)operations;
        (void)synchronizeAfter;
        return OpStatus::Unsupported;
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
