/* NCCL collective provider implementation.
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once

#include <alpaka/tensor/providers/EnabledVendorLibs.hpp>
#include <alpaka/tensor/providers/collective/CollectiveProviderInterface.hpp>

#include <cstring>
#include <optional>
#include <vector>

#ifdef ALPAKA_HAS_NCCL
#    include <cuda_runtime_api.h>
#    include <nccl.h>
#endif

namespace alpaka::tensor::collective
{
    namespace detail
    {
#ifdef ALPAKA_HAS_NCCL
        [[nodiscard]] inline std::optional<ncclDataType_t> toNccl(DataType type)
        {
            switch(type)
            {
            case DataType::Float32:
                return ncclFloat;
            case DataType::Float64:
                return ncclDouble;
            case DataType::Int32:
                return ncclInt32;
            case DataType::Int64:
                return ncclInt64;
            case DataType::UInt8:
                return ncclUint8;
            }
            return std::nullopt;
        }

        [[nodiscard]] inline std::optional<ncclRedOp_t> toNccl(ReduceOp op)
        {
            switch(op)
            {
            case ReduceOp::Sum:
                return ncclSum;
            case ReduceOp::Prod:
                return ncclProd;
            case ReduceOp::Min:
                return ncclMin;
            case ReduceOp::Max:
                return ncclMax;
            }
            return std::nullopt;
        }
#endif
    } // namespace detail

    class NcclCollectiveProvider : public ICollectiveProvider
    {
    public:
        NcclCollectiveProvider() = default;

        ~NcclCollectiveProvider() override
        {
#ifdef ALPAKA_HAS_NCCL
            for(auto& comm : comms_)
            {
                if(comm != nullptr)
                {
                    ncclCommDestroy(comm);
                    comm = nullptr;
                }
            }
#endif
        }

        std::string_view name() const override
        {
            return "NCCL";
        }

        bool isActive() const override
        {
            return active_;
        }

        std::size_t worldSize() const override
        {
            if(worldSizeOverride_ > 0)
                return worldSizeOverride_;
            return devices_.size();
        }

        [[nodiscard]] OpStatus initialize(GroupConfig const& config) override
        {
            reset();
            if(!EnabledVendorLibs::hasNCCL)
                return OpStatus::Unsupported;

#ifdef ALPAKA_HAS_NCCL
            if(config.multiProcess)
            {
                if(config.worldSize <= 0)
                {
                    reset();
                    return OpStatus::Error;
                }
                if(config.deviceIds.size() != 1u)
                {
                    reset();
                    return OpStatus::Error;
                }
                if(config.providerUniqueId.size() != sizeof(ncclUniqueId))
                {
                    reset();
                    return OpStatus::Error;
                }

                devices_ = config.deviceIds;
                comms_.resize(1u);
                ncclUniqueId id{};
                std::memcpy(&id, config.providerUniqueId.data(), sizeof(ncclUniqueId));

                if(cudaSetDevice(devices_.front()) != cudaSuccess)
                {
                    reset();
                    return OpStatus::Error;
                }

                auto const initStatus = ncclCommInitRank(&comms_.front(), config.worldSize, id, config.worldRank);
                if(initStatus != ncclSuccess)
                {
                    reset();
                    return OpStatus::Error;
                }

                worldSizeOverride_ = static_cast<std::size_t>(config.worldSize);
                active_ = true;
                return OpStatus::Success;
            }

            auto deviceIds = config.deviceIds;
            if(deviceIds.empty())
            {
                int count = 0;
                if(cudaGetDeviceCount(&count) != cudaSuccess || count <= 0)
                {
                    return OpStatus::Error;
                }
                deviceIds.resize(static_cast<std::size_t>(count));
                for(int i = 0; i < count; ++i)
                {
                    deviceIds[static_cast<std::size_t>(i)] = i;
                }
            }
            devices_ = deviceIds;
            comms_.resize(devices_.size());
            auto result = ncclCommInitAll(comms_.data(), static_cast<int>(devices_.size()), devices_.data());
            if(result != ncclSuccess)
            {
                reset();
                return OpStatus::Error;
            }
            active_ = true;
            return OpStatus::Success;
#else
            static_cast<void>(config);
            return OpStatus::Unsupported;
#endif
        }

        [[nodiscard]] OpStatus allReduce(AllReduceRequest const& request) override
        {
            if(!EnabledVendorLibs::hasNCCL)
                return OpStatus::Unsupported;
#ifdef ALPAKA_HAS_NCCL
            if(!active_ || devices_.empty())
                return OpStatus::Error;

            auto dtype = detail::toNccl(request.dataType);
            auto rop = detail::toNccl(request.reduceOp);
            if(!dtype || !rop)
                return OpStatus::Unsupported;

            auto const deviceCount = devices_.size();
            if(request.buffers.recv.size() != deviceCount)
                return OpStatus::Error;
            if(!request.buffers.inPlace && request.buffers.send.size() != deviceCount)
                return OpStatus::Error;
            if(!request.buffers.streams.empty() && request.buffers.streams.size() != deviceCount)
                return OpStatus::Error;

            auto getStream = [&](std::size_t idx) -> cudaStream_t
            {
                if(request.buffers.streams.empty() || request.buffers.streams[idx] == nullptr)
                    return nullptr;
                return reinterpret_cast<cudaStream_t>(request.buffers.streams[idx]);
            };

            auto getSendPtr = [&](std::size_t idx) -> void const*
            {
                if(request.buffers.inPlace)
                    return request.buffers.recv[idx];
                if(request.buffers.send.empty() || request.buffers.send[idx] == nullptr)
                    return request.buffers.recv[idx];
                return request.buffers.send[idx];
            };

            auto getRecvPtr = [&](std::size_t idx) -> void* { return request.buffers.recv[idx]; };

            if(deviceCount == 1u)
            {
                if(cudaSetDevice(devices_.front()) != cudaSuccess)
                    return OpStatus::Error;

                auto const sendPtr = getSendPtr(0u);
                auto const recvPtr = getRecvPtr(0u);
                auto const stream = getStream(0u);
                auto const status
                    = ncclAllReduce(sendPtr, recvPtr, request.elementCount, *dtype, *rop, comms_.front(), stream);
                if(status != ncclSuccess)
                    return OpStatus::Error;
                return OpStatus::Success;
            }

            ncclResult_t launchStatus = ncclGroupStart();
            if(launchStatus != ncclSuccess)
                return OpStatus::Error;

            ncclResult_t opStatus = ncclSuccess;
            for(std::size_t idx = 0; idx < deviceCount; ++idx)
            {
                if(cudaSetDevice(devices_[idx]) != cudaSuccess)
                {
                    opStatus = ncclSystemError;
                    break;
                }

                auto comm = comms_[idx];
                auto sendPtr = getSendPtr(idx);
                auto recvPtr = getRecvPtr(idx);
                auto stream = getStream(idx);
                opStatus = ncclAllReduce(sendPtr, recvPtr, request.elementCount, *dtype, *rop, comm, stream);
                if(opStatus != ncclSuccess)
                    break;
            }

            auto groupStatus = ncclGroupEnd();
            if(opStatus != ncclSuccess || groupStatus != ncclSuccess)
                return OpStatus::Error;

            return OpStatus::Success;
#else
            static_cast<void>(request);
            return OpStatus::Unsupported;
#endif
        }

    private:
        void reset()
        {
#ifdef ALPAKA_HAS_NCCL
            for(auto& comm : comms_)
            {
                if(comm != nullptr)
                {
                    ncclCommDestroy(comm);
                    comm = nullptr;
                }
            }
            comms_.clear();
#endif
            devices_.clear();
            active_ = false;
            worldSizeOverride_ = 0u;
        }

#ifdef ALPAKA_HAS_NCCL
        std::vector<ncclComm_t> comms_{};
#else
        std::vector<void*> comms_{};
#endif
        std::vector<int> devices_{};
        bool active_{false};
        std::size_t worldSizeOverride_{0u};
    };
} // namespace alpaka::tensor::collective
