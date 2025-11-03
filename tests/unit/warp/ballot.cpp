/* Copyright 2025 Mehmet Yusufoglu
 * SPDX-License-Identifier: MPL-2.0
 */

#include "utils.hpp"

#include <alpaka/onAcc/Warp.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <climits>
#include <cstdint>

using namespace alpaka;
using alpaka::test::warp::warpCheck;
using alpaka::test::warp::WarpTestBackends;

namespace
{
    struct BallotSingleThreadKernel
    {
        template<typename TAcc>
        ALPAKA_FN_ACC void operator()(TAcc const& acc, concepts::MdSpan<bool> auto success) const
        {
            warpCheck(success, onAcc::warp::getSize(acc) == 1u);
            warpCheck(success, onAcc::warp::ballot(acc, 42) == 1u);
            warpCheck(success, onAcc::warp::ballot(acc, 0) == 0u);
        }
    };

    struct BallotMultiThreadKernel
    {
        template<typename TAcc>
        ALPAKA_FN_ACC void operator()(TAcc const& acc, concepts::MdSpan<bool> auto success) const
        {
            auto const warpExtent = static_cast<std::uint32_t>(onAcc::warp::getSize(acc));
            warpCheck(success, warpExtent > 1);

            auto const threadsPerBlock = static_cast<std::uint32_t>(acc[alpaka::layer::thread].count().product());
            warpCheck(success, threadsPerBlock == warpExtent);

            using ResultType = decltype(onAcc::warp::ballot(acc, 42));
            auto const maxBits = static_cast<std::uint32_t>(sizeof(ResultType) * CHAR_BIT);
            auto const totalBits = std::min(warpExtent, maxBits);
            auto const allActive
                = totalBits == maxBits ? ~ResultType{0u} : (ResultType{1} << totalBits) - ResultType{1};

            warpCheck(success, onAcc::warp::ballot(acc, 42) == allActive);
            warpCheck(success, onAcc::warp::ballot(acc, 0) == 0u);

            auto const lane = static_cast<std::uint32_t>(onAcc::warp::getLaneIdx(acc));
            if(lane >= warpExtent / 2u)
            {
                return;
            }

            auto const activeLaneCount = static_cast<std::uint32_t>(warpExtent / 2u);
            for(std::uint32_t idx = 0u; idx < activeLaneCount; ++idx)
            {
                auto const bitMask = static_cast<ResultType>(ResultType{1} << idx);
                warpCheck(success, onAcc::warp::ballot(acc, lane == idx ? 1 : 0) == bitMask);

                auto const expected = ((ResultType{1} << activeLaneCount) - ResultType{1}) & ~bitMask;
                warpCheck(success, onAcc::warp::ballot(acc, lane == idx ? 0 : 1) == expected);
            }
        }
    };
} // namespace

TEMPLATE_LIST_TEST_CASE("warp ballot captures predicate lanes", "[warp][ballot]", WarpTestBackends)
{
    auto cfg = TestType::makeDict();
    auto deviceSpec = cfg[object::deviceSpec];
    auto exec = cfg[object::exec];

    auto selector = onHost::makeDeviceSelector(deviceSpec);
    if(!selector.isAvailable())
    {
        INFO("No device available for " << deviceSpec.getName());
        return;
    }

    auto device = selector.makeDevice(0);
    auto queue = device.makeQueue(queueKind::blocking);

    auto successHost = onHost::allocHost<bool>(1u);
    auto successDev = onHost::allocLike(device, successHost);

    auto const warpExtent = warp::getSize(deviceSpec.getApi(), deviceSpec.getDeviceKind());

    if(warpExtent == 1u)
    {
        onHost::memset(queue, successDev, static_cast<std::uint8_t>(true));
        queue.enqueue(
            exec,
            onHost::FrameSpec{Vec<std::uint32_t, 1u>{1u}, Vec<std::uint32_t, 1u>{1u}},
            KernelBundle{BallotSingleThreadKernel{}, successDev});
        onHost::memcpy(queue, successHost, successDev);
        onHost::wait(queue);
        CHECK(successHost[0]);
        return;
    }

    auto const blocks = Vec<std::uint32_t, 1u>{1u};
    auto const threads = Vec<std::uint32_t, 1u>{warpExtent};

    onHost::memset(queue, successDev, static_cast<std::uint8_t>(true));
    queue.enqueue(exec, onHost::FrameSpec{blocks, threads}, KernelBundle{BallotMultiThreadKernel{}, successDev});
    onHost::memcpy(queue, successHost, successDev);
    onHost::wait(queue);
    INFO("backend=" << deviceSpec.getName());
    CHECK(successHost[0]);
}
