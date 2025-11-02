/* Copyright 2025 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/Vec.hpp"
#include "alpaka/interface.hpp"
#include "alpaka/onAcc/Acc.hpp"
#include "alpaka/tag.hpp"
#include "alpaka/warp/Traits.hpp"

#include <cstdint>

namespace alpaka::onAcc::warp
{
    namespace detail
    {
        template<concepts::Acc T_Acc>
        ALPAKA_FN_HOST_ACC constexpr uint32_t warpSize(T_Acc const& acc)
        {
            return alpaka::warp::getSize(ALPAKA_TYPEOF(acc.getApi()){}, ALPAKA_TYPEOF(acc.getDeviceKind()){});
        }

        template<typename T_CountVec, typename T_IdxVec>
        ALPAKA_FN_HOST_ACC constexpr uint32_t linearThreadIdx(T_CountVec const& threadCount, T_IdxVec const& threadIdx)
        {
            auto const linear = linearize(threadCount, threadIdx);
            return static_cast<uint32_t>(linear);
        }
    } // namespace detail

    /** Return the number of lanes participating in a warp. */
    ALPAKA_FN_HOST_ACC constexpr uint32_t getSize(concepts::Acc auto const& acc)
    {
        return detail::warpSize(acc);
    }

    /** Return the lane index of the current thread within its warp. */
    ALPAKA_FN_HOST_ACC constexpr uint32_t getLaneIdx(concepts::Acc auto const& acc)
    {
        auto const& threadLayer = acc[alpaka::layer::thread];
        auto const linearIdx = detail::linearThreadIdx(threadLayer.count(), threadLayer.idx());
        auto const size = getSize(acc);
        return static_cast<uint32_t>(linearIdx % size);
    }

    /** Return the warp index of the current thread within the block. */
    ALPAKA_FN_HOST_ACC constexpr uint32_t getWarpIdxInBlock(concepts::Acc auto const& acc)
    {
        auto const& threadLayer = acc[alpaka::layer::thread];
        auto const linearIdx = detail::linearThreadIdx(threadLayer.count(), threadLayer.idx());
        auto const size = getSize(acc);
        return static_cast<uint32_t>(linearIdx / size);
    }

    /** Return the total number of warps required to cover the current block. */
    ALPAKA_FN_HOST_ACC constexpr uint32_t getNumWarps(concepts::Acc auto const& acc)
    {
        auto const& threadLayer = acc[alpaka::layer::thread];
        auto const totalThreads = static_cast<uint32_t>(threadLayer.count().product());
        auto const size = getSize(acc);
        return static_cast<uint32_t>((totalThreads + size - 1u) / size);
    }

    /** True if the current lane is the first lane within the warp. */
    ALPAKA_FN_HOST_ACC constexpr bool isWarpLeader(concepts::Acc auto const& acc)
    {
        return getLaneIdx(acc) == 0u;
    }

} // namespace alpaka::onAcc::warp

