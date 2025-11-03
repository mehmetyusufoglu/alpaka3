/* Copyright 2025 Mehmet Yusufoglu
 * SPDX-License-Identifier: MPL-2.0
 */

#include "utils.hpp"

#include <alpaka/onAcc/Warp.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <limits>

using namespace alpaka;
using alpaka::test::warp::warpCheck;
using alpaka::test::warp::WarpTestBackends;

namespace
{
    struct ShflUpSingleThreadKernel
    {
        template<typename TAcc>
        ALPAKA_FN_ACC void operator()(TAcc const& acc, concepts::MdSpan<bool> auto success) const
        {
            warpCheck(success, onAcc::warp::shflUp(acc, 42, 0u) == 42);
            warpCheck(success, onAcc::warp::shflUp(acc, 12, 0u) == 12);
            float const ans = onAcc::warp::shflUp(acc, 3.3f, 0u);
            warpCheck(success, ans == 3.3f);
        }
    };

    struct ShflUpMultiThreadKernel
    {
        template<typename TAcc>
        ALPAKA_FN_ACC void operator()(TAcc const& acc, concepts::MdSpan<bool> auto success) const
        {
            auto const warpExtent = static_cast<std::int32_t>(onAcc::warp::getSize(acc));
            warpCheck(success, warpExtent > 1);

            auto const threadsPerBlock = static_cast<std::int32_t>(acc[alpaka::layer::thread].count().product());
            warpCheck(success, threadsPerBlock == warpExtent);

            auto const lane = static_cast<std::int32_t>(onAcc::warp::getLaneIdx(acc));

            warpCheck(success, onAcc::warp::shflUp(acc, 42, 0u) == 42);
            warpCheck(success, onAcc::warp::shflUp(acc, lane, 0u) == lane);
            warpCheck(success, onAcc::warp::shflUp(acc, lane, 1u) == (lane - 1 >= 0 ? lane - 1 : lane));

            auto const epsilon = std::numeric_limits<float>::epsilon();
            for(int width = 1; width < warpExtent; width *= 2)
            {
                for(int idx = 0; idx < width; ++idx)
                {
                    auto const sectionStart = width * (lane / width);
                    auto const shuffled = onAcc::warp::shflUp(
                        acc,
                        lane,
                        static_cast<std::uint32_t>(idx),
                        static_cast<std::uint32_t>(width));
                    auto const expectedInt = (lane - idx >= sectionStart) ? lane - idx : lane;
                    warpCheck(success, shuffled == expectedInt);

                    auto const ans = onAcc::warp::shflUp(
                        acc,
                        4.0f - static_cast<float>(lane),
                        static_cast<std::uint32_t>(idx),
                        static_cast<std::uint32_t>(width));
                    auto const expect = (lane - idx >= sectionStart) ? 4.0f - static_cast<float>(lane - idx)
                                                                     : 4.0f - static_cast<float>(lane);
                    warpCheck(success, alpaka::math::abs(ans - expect) < epsilon);
                }
            }

            if(lane >= warpExtent / 2)
            {
                return;
            }

            for(int idx = 0; idx < warpExtent / 2; ++idx)
            {
                auto const shuffled = onAcc::warp::shflUp(acc, lane, static_cast<std::uint32_t>(idx));
                auto const ans
                    = onAcc::warp::shflUp(acc, 4.0f - static_cast<float>(lane), static_cast<std::uint32_t>(idx));
                auto const expect
                    = (lane - idx >= 0) ? 4.0f - static_cast<float>(lane - idx) : 4.0f - static_cast<float>(lane);
                warpCheck(success, shuffled == (lane - idx >= 0 ? lane - idx : lane));
                warpCheck(success, alpaka::math::abs(ans - expect) < epsilon);
            }
        }
    };
} // namespace

TEMPLATE_LIST_TEST_CASE("warp shflUp shifts toward lower lanes", "[warp][shfl_up]", WarpTestBackends)
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
            KernelBundle{ShflUpSingleThreadKernel{}, successDev});
        onHost::memcpy(queue, successHost, successDev);
        onHost::wait(queue);
        CHECK(successHost[0]);
        return;
    }

    auto const blocks = Vec<std::uint32_t, 1u>{1u};
    auto const threads = Vec<std::uint32_t, 1u>{warpExtent};

    onHost::memset(queue, successDev, static_cast<std::uint8_t>(true));
    queue.enqueue(exec, onHost::FrameSpec{blocks, threads}, KernelBundle{ShflUpMultiThreadKernel{}, successDev});
    onHost::memcpy(queue, successHost, successDev);
    onHost::wait(queue);
    INFO("backend=" << deviceSpec.getName());
    CHECK(successHost[0]);
}
