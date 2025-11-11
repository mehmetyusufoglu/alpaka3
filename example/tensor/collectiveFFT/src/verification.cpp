#include "collectiveFft/verification.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>

namespace collectiveFft
{
    bool compareSpectra(
        std::span<std::complex<float> const> computed,
        std::span<std::complex<float> const> reference,
        float absTolerance,
        float relTolerance,
        VerificationStats& stats)
    {
        stats = VerificationStats{};
        if(computed.size() != reference.size())
        {
            stats.failureCount = std::max(computed.size(), reference.size());
            stats.maxAbsError = std::numeric_limits<float>::infinity();
            stats.maxRelError = std::numeric_limits<float>::infinity();
            stats.worstIndex = 0;
            return false;
        }

        for(std::size_t idx = 0; idx < computed.size(); ++idx)
        {
            std::complex<float> const diff = computed[idx] - reference[idx];
            float const absDiff = std::abs(diff);
            float const referenceMag = std::abs(reference[idx]);
            float const relDiff = (referenceMag > 0.0f) ? absDiff / referenceMag : absDiff;

            if(absDiff > stats.maxAbsError)
            {
                stats.maxAbsError = absDiff;
                stats.worstIndex = idx;
            }
            if(relDiff > stats.maxRelError)
            {
                stats.maxRelError = relDiff;
            }
            if(absDiff > absTolerance && relDiff > relTolerance)
            {
                ++stats.failureCount;
            }
        }

        return stats.failureCount == 0;
    }

    namespace
    {
        template<typename Float>
        std::vector<std::complex<Float>> computeLocalFftGeneric(std::span<std::complex<Float> const> samples)
        {
            std::size_t const sampleCount = samples.size();
            std::vector<std::complex<Float>> spectrum(sampleCount, std::complex<Float>{0.0f, 0.0f});
            if(sampleCount == 0)
                return spectrum;

            constexpr Float twoPi = static_cast<Float>(2.0) * std::numbers::pi_v<Float>;
            for(std::size_t k = 0; k < sampleCount; ++k)
            {
                std::complex<Float> sum{0.0f, 0.0f};
                for(std::size_t n = 0; n < sampleCount; ++n)
                {
                    Float angle = -twoPi * static_cast<Float>(n * k) / static_cast<Float>(sampleCount);
                    sum += samples[n] * std::complex<Float>{std::cos(angle), std::sin(angle)};
                }
                spectrum[k] = sum;
            }

            return spectrum;
        }
    } // namespace

    std::vector<std::complex<float>> computeLocalFft(std::span<std::complex<float> const> samples)
    {
        return computeLocalFftGeneric<float>(samples);
    }

    std::vector<std::complex<float>> computeLocalFftWithPrecision(
        std::span<std::complex<float> const> samples,
        DirectVerifyPrecision precision)
    {
        if(precision == DirectVerifyPrecision::Float32)
        {
            return computeLocalFft(samples);
        }

        std::vector<std::complex<double>> doubleSamples(samples.size(), std::complex<double>{0.0, 0.0});
        for(std::size_t idx = 0; idx < samples.size(); ++idx)
        {
            doubleSamples[idx] = std::complex<double>{
                static_cast<double>(samples[idx].real()),
                static_cast<double>(samples[idx].imag())};
        }

        auto doubleSpectrum = computeLocalFftGeneric<double>(
            std::span<std::complex<double> const>{doubleSamples.data(), doubleSamples.size()});

        std::vector<std::complex<float>> spectrum(doubleSpectrum.size(), std::complex<float>{0.0f, 0.0f});
        for(std::size_t idx = 0; idx < doubleSpectrum.size(); ++idx)
        {
            spectrum[idx] = std::complex<float>{
                static_cast<float>(doubleSpectrum[idx].real()),
                static_cast<float>(doubleSpectrum[idx].imag())};
        }

        return spectrum;
    }
} // namespace collectiveFft
