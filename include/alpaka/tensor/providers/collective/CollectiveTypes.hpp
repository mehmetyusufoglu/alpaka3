/* Collective communication type system used by tensor providers.
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once

#include <cstddef>
#include <span>
#include <vector>

namespace alpaka::tensor::collective
{
    // Enumerates the scalar element types we can ship through a collective.
    enum class DataType
    {
        Float32,
        Float64,
        Int32,
        Int64,
        UInt8
    };

    // Reduction operators supported by the collective providers.
    enum class ReduceOp
    {
        Sum,
        Prod,
        Min,
        Max
    };

    // High-level collective operations surfaced to tensor providers.
    enum class Operation
    {
        AllReduce,
        Broadcast,
        AllGather,
        ReduceScatter,
        Barrier
    };

    /**
     * GroupConfig describes the logical communicator we expect the provider to realize.
     * The device IDs identify which local accelerators participate; worldRank/worldSize encode
     * the MPI view when multiProcess is true; providerUniqueId carries NCCL/other backend tokens
     * that allow processes to rendezvous outside of Alpaka.
     */
    struct GroupConfig
    {
        std::vector<int> deviceIds{};
        bool multiProcess = false;
        int worldRank = 0;
        int worldSize = 0;
        std::vector<std::byte> providerUniqueId{};
    };

    /**
     * MultiDeviceBuffers bundles raw device pointers and streams for a collective invocation.
     * Providers expect matching spans for send/recv/streams; inPlace allows us to reuse the same
     * buffer for input and output when the backend supports it.
     */
    struct MultiDeviceBuffers
    {
        std::span<void const*> send = {};
        std::span<void*> recv = {};
        std::span<void*> streams = {};
        bool inPlace = false;
    };

    /**
     * AllReduceRequest instructs the provider to perform an elementwise reduction across all
     * ranks/devices in the group using the supplied buffers and operator.
     */
    struct AllReduceRequest
    {
        MultiDeviceBuffers buffers;
        std::size_t elementCount = 0;
        DataType dataType = DataType::Float32;
        ReduceOp reduceOp = ReduceOp::Sum;
    };

    /**
     * BroadcastRequest sends the buffer owned by rootRank to every other participant.
     */
    struct BroadcastRequest
    {
        MultiDeviceBuffers buffers;
        std::size_t elementCount = 0;
        DataType dataType = DataType::Float32;
        int rootRank = 0;
    };

    // BarrierRequest lets the provider synchronize streams without transferring payload data.
    struct BarrierRequest
    {
        std::span<void*> streams{};
    };
} // namespace alpaka::tensor::collective
