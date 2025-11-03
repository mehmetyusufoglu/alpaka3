/* Copyright 2025 Mehmet Yusufoglu
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <alpaka/alpaka.hpp>
#include <alpaka/onHost/example/executors.hpp>

#include <algorithm>
#include <cstdint>
#include <limits>

namespace alpaka::test::warp
{
    using WarpTestBackends
        = std::decay_t<decltype(onHost::allBackends(onHost::enabledApis, onHost::example::enabledExecutors))>;

    template<typename SuccessView>
    ALPAKA_FN_HOST_ACC inline void warpCheck(SuccessView success, bool condition)
    {
        if(!condition)
        {
            success[0u] = false;
        }
    }

    inline std::uint64_t fullMask(std::uint32_t warpSize)
    {
        if(warpSize == 0u)
        {
            return 0u;
        }
        if(warpSize >= 64u)
        {
            return std::numeric_limits<std::uint64_t>::max();
        }
        return (std::uint64_t{1} << warpSize) - 1u;
    }

    inline std::uint64_t evenMask(std::uint32_t warpSize)
    {
        auto const limit = std::min<std::uint32_t>(warpSize, 64u);
        std::uint64_t mask = 0u;
        for(std::uint32_t lane = 0u; lane < limit; lane += 2u)
        {
            mask |= (std::uint64_t{1} << lane);
        }
        return mask;
    }

    inline std::uint64_t singleBit(std::uint32_t lane)
    {
        if(lane >= 64u)
        {
            return 0u;
        }
        return std::uint64_t{1} << lane;
    }

    template<typename T_Api, typename T_DeviceKind>
    consteval std::uint32_t compileTimeWarpSize(T_Api const api, T_DeviceKind const device)
    {
        return alpaka::warp::getSize(api, device);
    }
} // namespace alpaka::test::warp
