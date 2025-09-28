/* NCCL Collective Provider
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once

#include <alpaka/tensor/ops/CollectiveOps.hpp>
#include <alpaka/tensor/providers/ICollectiveProvider.hpp>

#include <stdexcept>
#include <string>
#include <vector>

#ifdef ALPAKA_HAS_NCCL
#    include <cuda_runtime_api.h>
#    include <nccl.h>
#endif

namespace alpaka::tensor
{
    class NCCLProvider : public ICollectiveProvider
    {
    public:
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

        Diagnostics diagnostics() const;

        bool supportsPattern(ops::CollectivePattern pattern) const override;
        bool supportsReduction(ops::CollectiveReduction reduction) const override;
        bool supportsDataType(ops::CollectiveDataType dtype) const override;

        std::size_t worldSize() const override;
        std::size_t worldRank() const override;

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

#ifdef ALPAKA_HAS_NCCL
        ncclDataType_t mapDataType(ops::CollectiveDataType dtype) const;
        ncclRedOp_t mapReduction(ops::CollectiveReduction reduction) const;
#endif

    private:
#ifdef ALPAKA_HAS_NCCL
        mutable ncclComm_t comm_ = nullptr;
        mutable bool initialized_ = false;
        mutable bool active_ = false;
        mutable int worldSize_ = 1;
        mutable int worldRank_ = 0;
#else
        static constexpr bool active_ = false;
#endif
    };
} // namespace alpaka::tensor

#ifdef ALPAKA_HAS_NCCL
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
        if(initialized_)
            return;

        int deviceId = 0;
        cudaError_t cudaStatus = cudaGetDevice(&deviceId);
        if(cudaStatus != cudaSuccess)
        {
            initialized_ = true;
            active_ = false;
            return;
        }

        ncclResult_t ncclStatus = ncclCommInitAll(&comm_, 1, &deviceId);
        if(ncclStatus != ncclSuccess)
        {
            initialized_ = true;
            active_ = false;
            comm_ = nullptr;
            return;
        }

        int size = 0;
        if(ncclCommCount(comm_, &size) == ncclSuccess && size > 0)
        {
            worldSize_ = size;
        }
        else
        {
            worldSize_ = 1;
        }

        int rank = 0;
        if(ncclCommUserRank(comm_, &rank) == ncclSuccess)
        {
            worldRank_ = rank;
        }
        else
        {
            worldRank_ = 0;
        }

        active_ = true;
        initialized_ = true;
    }

    inline void NCCLProvider::finalize() noexcept
    {
        if(comm_ != nullptr)
        {
            ncclCommDestroy(comm_);
            comm_ = nullptr;
        }
        initialized_ = false;
        active_ = false;
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

        cudaStream_t cudaStream = ctx.nativeQueue ? static_cast<cudaStream_t>(ctx.nativeQueue) : nullptr;
        ncclResult_t status = ncclAllReduce(
            sendBuffer,
            recvBuffer,
            elementCount,
            mapDataType(dtype),
            mapReduction(reduction),
            comm_,
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

        cudaStream_t cudaStream = ctx.nativeQueue ? static_cast<cudaStream_t>(ctx.nativeQueue) : nullptr;
        ncclResult_t status = ncclBroadcast(
            static_cast<void const*>(buffer),
            buffer,
            elementCount,
            mapDataType(dtype),
            static_cast<int>(rootRank),
            comm_,
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
