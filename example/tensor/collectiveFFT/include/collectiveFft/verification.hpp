#pragma once

#include "collectiveFft/options.hpp"

#include <complex>
#include <span>
#include <vector>

namespace collectiveFft
{
    struct VerificationStats
    {
        float maxAbsError = 0.0f;
        float maxRelError = 0.0f;
        std::size_t worstIndex = 0;
        std::size_t failureCount = 0;
    };

    bool compareSpectra(
        std::span<std::complex<float> const> computed,
        std::span<std::complex<float> const> reference,
        float absTolerance,
        float relTolerance,
        VerificationStats& stats);

    std::vector<std::complex<float>> computeLocalFft(std::span<std::complex<float> const> samples);

    std::vector<std::complex<float>> computeLocalFftWithPrecision(
        std::span<std::complex<float> const> samples,
        DirectVerifyPrecision precision);
} // namespace collectiveFft
