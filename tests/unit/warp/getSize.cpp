/* Copyright 2025 Mehmet Yusufoglu
 * SPDX-License-Identifier: MPL-2.0
 */

#include "utils.hpp"

#include <alpaka/onAcc/Warp.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstdint>

using namespace alpaka;
using alpaka::test::warp::warpCheck;
using alpaka::test::warp::WarpTestBackends;

namespace
{
    struct GetSizeKernel
    {
        template<typename TAcc>
        ALPAKA_FN_ACC void operator()(
            TAcc const& acc,
            concepts::MdSpan<bool> auto success,
            std::uint32_t expectedWarpSize) const
        {
            auto const runtimeSize = onAcc::warp::getSize(acc);
            warpCheck(success, runtimeSize != 0u);
            warpCheck(success, runtimeSize == expectedWarpSize);

            using ApiTag = ALPAKA_TYPEOF(acc.getApi());
            using DeviceTag = ALPAKA_TYPEOF(acc.getDeviceKind());
            constexpr auto compileTimeSize = alpaka::test::warp::compileTimeWarpSize(ApiTag{}, DeviceTag{});
            warpCheck(success, runtimeSize == compileTimeSize);
        }
    };
} // namespace

TEMPLATE_LIST_TEST_CASE("warp size trait matches runtime size", "[warp][getSize]", WarpTestBackends)
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
    auto const blocks = Vec<std::uint32_t, 1u>{1u};
    auto const threads = Vec<std::uint32_t, 1u>{warpExtent == 0u ? 1u : warpExtent};

    onHost::memset(queue, successDev, static_cast<std::uint8_t>(true));
    queue.enqueue(
        exec,
        onHost::FrameSpec{blocks, threads},
        KernelBundle{GetSizeKernel{}, successDev, static_cast<std::uint32_t>(warpExtent)});
    onHost::memcpy(queue, successHost, successDev);
    onHost::wait(queue);
    INFO("backend=" << deviceSpec.getName());
    CHECK(successHost[0]);
}
