/* Transformer FFN Stage 2 Kernel: Hacc[M,4H] * W2[4H,H] + b2[H] -> Y[M,H] */
#pragma once

#include <alpaka/alpaka.hpp>

struct FfnStage2Kernel
{
    template<typename TAcc, typename T>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc,
                                  alpaka::concepts::MdSpan auto const h,   // [M,4H]
                                  alpaka::concepts::MdSpan auto const w2,  // [4H,H]
                                  alpaka::concepts::MdSpan auto y,         // [M,H]
                                  T const* __restrict__ b2,                // [H]
                                  std::size_t M, std::size_t H, std::size_t fourH) const -> void
    {
        using namespace alpaka;
        for(concepts::Dim<2u> auto idx2d :
            onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, IdxRange{Vec{M, H}}))
        {
            auto const m = static_cast<std::size_t>(idx2d[0]);
            auto const j = static_cast<std::size_t>(idx2d[1]);
            if(m >= M || j >= H) continue;

            float accVal = 0.f;
            for(std::size_t k = 0; k < fourH; ++k)
            {
                auto idxH  = Vec{m, k};
                auto idxW2 = Vec{k, j};
                accVal += static_cast<float>(h[idxH]) * static_cast<float>(w2[idxW2]);
            }
            accVal += static_cast<float>(b2[j]);
            y[idx2d] = static_cast<T>(accVal);
        }
    }
};
