/* Transformer FFN Stage 1 Kernel (GEMM + Bias + GELU)
 * Shape: X[M,H] * W1[H,4H] + b1[4H] -> Hacc[M,4H]
 * Reduced precision friendly; accumulation in float for stability.
 */
#pragma once

#include <alpaka/alpaka.hpp>
#include <cmath>

struct FfnStage1Kernel
{
    template<typename TAcc, typename T>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc,
                                  alpaka::concepts::MdSpan auto const x,   // [M,H]
                                  alpaka::concepts::MdSpan auto const w1,  // [H,4H]
                                  alpaka::concepts::MdSpan auto h,         // [M,4H]
                                  T const* __restrict__ b1,                // [4H]
                                  std::size_t M, std::size_t H, std::size_t fourH) const -> void
    {
        using namespace alpaka;

        for(concepts::Dim<2u> auto idx2d :
            onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, IdxRange{Vec{M, fourH}}))
        {
            auto const m = static_cast<std::size_t>(idx2d[0]);
            auto const j = static_cast<std::size_t>(idx2d[1]);
            if(m >= M || j >= fourH) continue;

            float accVal = 0.0f;
            for(std::size_t k = 0; k < H; ++k)
            {
                auto idxX  = Vec{m, k};
                auto idxW1 = Vec{k, j};
                accVal += static_cast<float>(x[idxX]) * static_cast<float>(w1[idxW1]);
            }
            accVal += static_cast<float>(b1[j]);
            // Fast GELU approximation (as in many MLPerf kernels)
            float const u = accVal;
            float const gelu = 0.5f * u * (1.f + tanhf(0.79788456f * (u + 0.044715f * u * u * u)));
            h[idx2d] = static_cast<T>(gelu);
        }
    }
};
