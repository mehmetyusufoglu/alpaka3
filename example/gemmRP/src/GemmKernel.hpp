/* Copyright 2024 Alpaka Group
 * SPDX-License-Identifier: ISC
 */

#pragma once

#include <alpaka/alpaka.hpp>

//! GEMM kernel for demonstrating reduced precision types
//! C = alpha * A * B + beta * C
struct GemmKernel
{
    template<typename TAcc, typename ValueType>
    ALPAKA_FN_ACC auto operator()(
        TAcc const& acc,
        alpaka::concepts::MdSpan auto const matrixA,
        alpaka::concepts::MdSpan auto const matrixB,
        alpaka::concepts::MdSpan auto matrixC,
        std::size_t M,
        std::size_t N,
        std::size_t K,
        ValueType alpha,
        ValueType beta) const -> void
    {
        using namespace alpaka;

        // Iterate over all matrix elements
        for(concepts::Dim<2u> auto idx2d :
            onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, IdxRange{Vec{M, N}}))
        {
            auto const i = static_cast<std::size_t>(idx2d[0]); // row
            auto const j = static_cast<std::size_t>(idx2d[1]); // col

            ValueType accVal = ValueType(0);
            
            // Compute dot product of row i of A with column j of B
            for(std::size_t k = 0; k < K; ++k)
            {
                auto idxA = Vec{i, k};
                auto idxB = Vec{k, j};
                accVal += matrixA[idxA] * matrixB[idxB];
            }
            
            // C = alpha * A * B + beta * C
            matrixC[idx2d] = alpha * accVal + beta * matrixC[idx2d];
        }
    }
};
