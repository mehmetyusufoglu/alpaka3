/* SPDX-License-Identifier: MPL-2.0 */

#pragma once

#include "alpaka/Vec.hpp"
#include "alpaka/api/host/IdxLayer.hpp"
#include "alpaka/api/host/block/mem/SingleThreadStaticShared.hpp"
#include "alpaka/api/host/block/sync/NoOp.hpp"
#include "alpaka/core/Dict.hpp"
#include "alpaka/meta/NdLoop.hpp"
#include "alpaka/onAcc/Acc.hpp"
#include "alpaka/onHost/ThreadSpec.hpp"
#include "alpaka/tag.hpp"

#include <cstddef>
#include <stdexcept>
#include <tuple>

#if ALPAKA_TBB
#    include <oneapi/tbb/blocked_range.h>
#    include <oneapi/tbb/enumerable_thread_specific.h>
#    include <oneapi/tbb/parallel_for.h>

namespace alpaka::onHost
{
    namespace cpu
    {
        template<onHost::concepts::ThreadSpec T_ThreadSpec>
        struct TbbBlocks
        {
            using NumThreadsVecType = typename T_ThreadSpec::NumThreadsVecType;

            constexpr TbbBlocks(T_ThreadSpec threadBlocking) : m_threadBlocking(std::move(threadBlocking))
            {
            }

            void operator()(auto const& kernelBundle, auto const& dict) const
            {
                if(m_threadBlocking.m_numThreads.product() != 1u)
                    throw std::runtime_error("Thread block extent must be 1.");

                auto const blockCount = m_threadBlocking.m_numBlocks;
                constexpr uint32_t simdWidth
                    = alpaka::getArchSimdWidth<uint8_t>(api::host, ALPAKA_TYPEOF(dict[object::deviceKind]){});
                using SharedStorage = onAcc::cpu::SingleThreadStaticShared<simdWidth>;

                oneapi::tbb::enumerable_thread_specific<SharedStorage> sharedMemTLS;

                using ThreadIdxType = typename NumThreadsVecType::type;

                oneapi::tbb::parallel_for(
                    oneapi::tbb::blocked_range<std::size_t>(0u, static_cast<std::size_t>(blockCount.product())),
                    [&](oneapi::tbb::blocked_range<std::size_t> const& range)
                    {
                        auto blockIdx = blockCount;
                        auto& blockSharedMem = sharedMemTLS.local();

                        auto const blockLayerEntry = DictEntry{
                            layer::block,
                            onAcc::cpu::GenericLayer{std::cref(blockIdx), std::cref(blockCount)}};
                        auto const threadLayerEntry
                            = DictEntry{layer::thread, onAcc::cpu::OneLayer<NumThreadsVecType>{}};
                        auto const blockSharedMemEntry = DictEntry{layer::shared, std::ref(blockSharedMem)};
                        auto const blockSyncEntry = DictEntry{action::threadBlockSync, onAcc::cpu::NoOp{}};

                        uint32_t blockDynSharedMemBytes
                            = onHost::getDynSharedMemBytes(exec::CpuTbbBlocks{}, m_threadBlocking, kernelBundle);
                        auto const blockDynSharedMemEntry = DictEntry{layer::dynShared, std::ref(blockSharedMem)};
                        auto const blockDynSharedMemBytesEntry
                            = DictEntry{object::dynSharedMemBytes, std::ref(blockDynSharedMemBytes)};

                        auto additionalDict = conditionalAppendDict<trait::HasUserDefinedDynSharedMemBytes<
                            exec::CpuTbbBlocks,
                            T_ThreadSpec,
                            ALPAKA_TYPEOF(kernelBundle)>::value>(
                            dict,
                            Dict{blockDynSharedMemEntry, blockDynSharedMemBytesEntry});

                        auto acc = onAcc::Acc(joinDict(
                            Dict{blockLayerEntry, threadLayerEntry, blockSharedMemEntry, blockSyncEntry},
                            additionalDict));

                        for(std::size_t i = range.begin(); i < range.end(); ++i)
                        {
                            blockIdx = mapToND(blockCount, static_cast<ThreadIdxType>(i));
                            kernelBundle(acc);
                            blockSharedMem.reset();
                        }
                    });
            }

            T_ThreadSpec m_threadBlocking;
        };
    } // namespace cpu

    inline auto makeAcc(exec::CpuTbbBlocks, auto const& threadBlocking)
    {
        return cpu::TbbBlocks(threadBlocking);
    }
} // namespace alpaka::onHost

#else

namespace alpaka::onHost
{
    template<typename T_ThreadSpec>
    auto makeAcc(exec::CpuTbbBlocks, T_ThreadSpec const&) = delete;
} // namespace alpaka::onHost

#endif
