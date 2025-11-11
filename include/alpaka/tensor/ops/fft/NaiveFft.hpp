#pragma once

#include <alpaka/tensor/providers/ProviderInterface.hpp>

#include <array>
#include <cmath>
#include <complex>
#include <numbers>
#include <span>
#include <stdexcept>
#include <type_traits>

namespace alpaka::tensor::ops::fft
{
    namespace detail
    {
        inline std::array<std::size_t, 3> computeStrides(std::array<std::size_t, 3> const& lengths, unsigned int rank)
        {
            std::array<std::size_t, 3> strides{1, 1, 1};
            if(rank == 0)
                return strides;

            strides[rank - 1U] = 1;
            if(rank >= 2U)
            {
                for(int idx = static_cast<int>(rank) - 2; idx >= 0; --idx)
                {
                    auto const next = static_cast<std::size_t>(idx) + 1U;
                    strides[static_cast<std::size_t>(idx)] = strides[next] * lengths[next];
                }
            }
            return strides;
        }

        inline std::array<std::size_t, 3> decodeIndex(
            std::size_t linearIndex,
            std::array<std::size_t, 3> const& lengths,
            std::array<std::size_t, 3> const& strides,
            unsigned int rank)
        {
            std::array<std::size_t, 3> coords{0, 0, 0};
            for(unsigned int dim = 0; dim < rank; ++dim)
            {
                auto const stride = strides[dim];
                coords[dim] = stride > 0 ? (linearIndex / stride) % lengths[dim] : 0U;
            }
            return coords;
        }
    } // namespace detail

    template<typename ComplexT>
    void naiveNdFft(std::span<ComplexT const> input, std::span<ComplexT> output, ops::FftParams const& params)
    {
        static_assert(
            std::is_same_v<ComplexT, std::complex<float>> || std::is_same_v<ComplexT, std::complex<double>>,
            "Naive FFT fallback currently supports float32/float64 complex types only");

        if(params.transformType != ops::FftTransformType::ComplexToComplex)
        {
            throw std::invalid_argument("Naive FFT fallback only supports complex-to-complex transforms");
        }

        unsigned int const rank = params.rank == 0U ? 1U : std::min<unsigned int>(params.rank, 3U);
        std::array<std::size_t, 3> lengths = params.lengths;
        for(unsigned int dim = rank; dim < 3U; ++dim)
        {
            lengths[dim] = 1U;
        }

        std::size_t elementsPerBatch = 1;
        for(unsigned int dim = 0; dim < rank; ++dim)
        {
            if(lengths[dim] == 0U)
            {
                throw std::invalid_argument("FFT dimension length must be greater than zero");
            }
            elementsPerBatch *= lengths[dim];
        }

        std::size_t const totalElements = elementsPerBatch * params.batch;
        if(input.size() != totalElements || output.size() != totalElements)
        {
            throw std::invalid_argument("Input/output span size mismatch for FFT fallback");
        }

        auto const strides = detail::computeStrides(lengths, rank);
        using Value = typename ComplexT::value_type;
        double const sign = params.direction == ops::FftDirection::Forward ? -1.0 : 1.0;
        double const twoPi = 2.0 * std::numbers::pi_v<double>;

        for(std::size_t batch = 0; batch < params.batch; ++batch)
        {
            auto const* batchInput = input.data() + batch * elementsPerBatch;
            auto* batchOutput = output.data() + batch * elementsPerBatch;

            for(std::size_t outIdx = 0; outIdx < elementsPerBatch; ++outIdx)
            {
                auto const outCoord = detail::decodeIndex(outIdx, lengths, strides, rank);
                ComplexT sum{Value{0}, Value{0}};

                for(std::size_t inIdx = 0; inIdx < elementsPerBatch; ++inIdx)
                {
                    auto const inCoord = detail::decodeIndex(inIdx, lengths, strides, rank);
                    double phaseAccum = 0.0;
                    for(unsigned int dim = 0; dim < rank; ++dim)
                    {
                        phaseAccum += static_cast<double>(outCoord[dim]) * static_cast<double>(inCoord[dim])
                                      / static_cast<double>(lengths[dim]);
                    }
                    double const phase = sign * twoPi * phaseAccum;
                    double const c = std::cos(phase);
                    double const s = std::sin(phase);
                    ComplexT const twiddle{static_cast<Value>(c), static_cast<Value>(s)};
                    sum += batchInput[inIdx] * twiddle;
                }

                batchOutput[outIdx] = sum;
            }
        }
    }
} // namespace alpaka::tensor::ops::fft
