/* Copyright 2025 Mehmet Yusufoglu
 * SPDX-License-Identifier: MPL-2.0
 */

#include "utils.hpp"

#include <alpaka/onAcc/Warp.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstdint>

using namespace alpaka;
using alpaka::test::warp::fullMask;
using alpaka::test::warp::singleBit;
using alpaka::test::warp::warpCheck;
using alpaka::test::warp::WarpTestBackends;

namespace
{
    struct ActivemaskSingleThreadKernel
    {
        template<typename TAcc>
        ALPAKA_FN_ACC void operator()(TAcc const& acc, concepts::MdSpan<bool> auto success) const
        {
            warpCheck(success, onAcc::warp::getSize(acc) == 1u);
            warpCheck(success, onAcc::warp::activemask(acc) == 1u);
        }
    };

    struct ActivemaskMultiThreadKernel
    {
        template<typename TAcc>
        ALPAKA_FN_ACC void operator()(TAcc const& acc, concepts::MdSpan<bool> auto success, std::uint32_t inactiveLane)
            const
        {
            auto const warpExtent = static_cast<std::uint32_t>(onAcc::warp::getSize(acc));
            warpCheck(success, warpExtent > 1u);

            auto const threadsPerBlock = static_cast<std::uint32_t>(acc[alpaka::layer::thread].count().product());
            warpCheck(success, threadsPerBlock == warpExtent);

            auto const lane = static_cast<std::uint32_t>(onAcc::warp::getLaneIdx(acc));
            if(lane == inactiveLane)
            {
                return;
            }

            auto const mask = onAcc::warp::activemask(acc);
            auto const expected = fullMask(warpExtent) & ~singleBit(inactiveLane);
            warpCheck(success, mask == expected);
        }
    };
} // namespace

TEMPLATE_LIST_TEST_CASE("warp activemask reflects participating lanes", "[warp][activemask]", WarpTestBackends)
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
            KernelBundle{ActivemaskSingleThreadKernel{}, successDev});
        onHost::memcpy(queue, successHost, successDev);
        onHost::wait(queue);
        CHECK(successHost[0]);
        return;
    }

    auto const blocks = Vec<std::uint32_t, 1u>{1u};
    auto const threads = Vec<std::uint32_t, 1u>{warpExtent};
    auto const frame = onHost::FrameSpec{blocks, threads};

    for(std::uint32_t inactiveLane = 0u; inactiveLane < warpExtent; ++inactiveLane)
    {
        onHost::memset(queue, successDev, static_cast<std::uint8_t>(true));
        queue.enqueue(exec, frame, KernelBundle{ActivemaskMultiThreadKernel{}, successDev, inactiveLane});
        onHost::memcpy(queue, successHost, successDev);
        onHost::wait(queue);
        INFO("backend=" << deviceSpec.getName() << " inactiveLane=" << inactiveLane);
        CHECK(successHost[0]);
    }
}
