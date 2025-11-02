/* Copyright 2025 Mehmet Yusufoglu, René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>

#include <catch2/catch_test_macros.hpp>

namespace
{
    template<typename VecType>
    struct ThreadLayer
    {
        VecType idxValue;
        VecType countValue;

        constexpr VecType idx() const
        {
            return idxValue;
        }

        constexpr VecType count() const
        {
            return countValue;
        }
    };

    template<typename ApiTag, typename DeviceKindTag, typename VecType>
    auto makeTestAcc(ApiTag api, DeviceKindTag deviceKind, VecType threadIdx, VecType threadCount)
    {
        return alpaka::onAcc::Acc{alpaka::Dict{
            alpaka::DictEntry{alpaka::layer::thread, ThreadLayer<VecType>{threadIdx, threadCount}},
            alpaka::DictEntry{alpaka::object::api, api},
            alpaka::DictEntry{alpaka::object::deviceKind, deviceKind}}};
    }
} // namespace

using namespace alpaka;

static_assert(getWarpSize(api::host, deviceKind::cpu) == 1u);
static_assert(getWarpSize(api::cuda, deviceKind::nvidiaGpu) == 32u);
static_assert(getWarpSize(api::hip, deviceKind::amdGpu) == 64u);
static_assert(getWarpSize(api::oneApi, deviceKind::intelGpu) == 32u);

TEST_CASE("warp lane arithmetic on host", "[warp]")
{
    auto threadIdx = Vec<uint32_t, 1u>{3u};
    auto threadCount = Vec<uint32_t, 1u>{8u};
    auto acc = makeTestAcc(api::host, deviceKind::cpu, threadIdx, threadCount);

    REQUIRE(onAcc::warp::getSize(acc) == 1u);
    REQUIRE(onAcc::warp::getLaneIdx(acc) == 0u);
    REQUIRE(onAcc::warp::getWarpIdxInBlock(acc) == 3u);
    REQUIRE(onAcc::warp::getNumWarps(acc) == 8u);
    REQUIRE(onAcc::warp::isWarpLeader(acc));
}

TEST_CASE("warp lane arithmetic on simulated gpu", "[warp]")
{
    auto threadIdx = Vec<uint32_t, 1u>{37u};
    auto threadCount = Vec<uint32_t, 1u>{128u};
    auto acc = makeTestAcc(api::cuda, deviceKind::nvidiaGpu, threadIdx, threadCount);

    REQUIRE(onAcc::warp::getSize(acc) == 32u);
    REQUIRE(onAcc::warp::getLaneIdx(acc) == 5u);
    REQUIRE(onAcc::warp::getWarpIdxInBlock(acc) == 1u);
    REQUIRE(onAcc::warp::getNumWarps(acc) == 4u);
    CHECK_FALSE(onAcc::warp::isWarpLeader(acc));
}

TEST_CASE("warp single thread vote and shuffle behaviour", "[warp]")
{
    constexpr auto apiTag = api::host;
    constexpr auto deviceTag = deviceKind::cpu;

    CHECK(alpaka::warp::activemask(apiTag, deviceTag) == 1u);
    CHECK(alpaka::warp::all(apiTag, deviceTag, true));
    CHECK_FALSE(alpaka::warp::all(apiTag, deviceTag, false));

    CHECK(alpaka::warp::any(apiTag, deviceTag, true));
    CHECK_FALSE(alpaka::warp::any(apiTag, deviceTag, false));

    CHECK(alpaka::warp::ballot(apiTag, deviceTag, true) == 1u);
    CHECK(alpaka::warp::ballot(apiTag, deviceTag, false) == 0u);

    constexpr int intValue = 42;
    CHECK(alpaka::warp::shfl(apiTag, deviceTag, intValue, 0u, 1u) == intValue);
    CHECK(alpaka::warp::shflDown(apiTag, deviceTag, intValue, 1u, 1u) == intValue);
    CHECK(alpaka::warp::shflUp(apiTag, deviceTag, intValue, 1u, 1u) == intValue);
    CHECK(alpaka::warp::shflXor(apiTag, deviceTag, intValue, 1u, 1u) == intValue);

    constexpr float floatValue = 3.5f;
    CHECK(alpaka::warp::shfl(apiTag, deviceTag, floatValue, 0u, 1u) == floatValue);
}
