#include "collectiveFft/utilities.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <iterator>

namespace
{
    [[nodiscard]] std::optional<int> parseEnvInt(char const* envName)
    {
        if(char const* value = std::getenv(envName))
        {
            char* end = nullptr;
            long parsed = std::strtol(value, &end, 10);
            if(end != nullptr && end != value && *end == '\0' && parsed >= 0)
            {
                return static_cast<int>(parsed);
            }
        }
        return std::nullopt;
    }
} // namespace

namespace collectiveFft::detail
{
    [[nodiscard]] std::optional<int> discoverLocalRank(MultiProcessBootstrap const& bootstrap)
    {
        if(bootstrap.localRank >= 0)
            return bootstrap.localRank;

        static constexpr std::array<char const*, 4> envKeys{
            "OMPI_COMM_WORLD_LOCAL_RANK",
            "MPI_LOCALRANKID",
            "SLURM_LOCALID",
            "LOCAL_RANK"};
        for(auto const* key : envKeys)
        {
            if(auto const value = parseEnvInt(key))
                return value;
        }
        return std::nullopt;
    }

    void buildContributionTile(
        std::span<std::complex<float> const> localSpectrum,
        int worldRank,
        std::size_t worldSize,
        std::size_t totalSamples,
        std::size_t tileOffset,
        std::span<std::complex<float>> tileBuffer)
    {
        (void) worldSize;
        std::size_t const samplesPerRank = localSpectrum.size();
        if(samplesPerRank == 0 || totalSamples == 0 || tileBuffer.empty())
            return;

        // Each tile maps to a contiguous range [tileOffset, tileOffset + tileBuffer.size()).
        // We only touch bins inside the global domain; callers supply offsets that honour bounds.
        constexpr double twoPi = 6.283185307179586476925286766559;
        double const totalSamplesAsDouble = static_cast<double>(totalSamples);
        for(std::size_t localIdx = 0; localIdx < tileBuffer.size(); ++localIdx)
        {
            std::size_t const globalBin = tileOffset + localIdx;
            if(globalBin >= totalSamples)
                break;

            // Strided ownership: bin k depends on the rank's spectrum entry at k mod samplesPerRank.
            std::size_t const spectrumIndex = globalBin % samplesPerRank;
            auto const localValue = localSpectrum[spectrumIndex];

            // Phase shift encodes the offset introduced by taking every worldSize-th sample.
            double const angle
                = -twoPi * static_cast<double>(worldRank) * static_cast<double>(globalBin) / totalSamplesAsDouble;
            float const cosAngle = static_cast<float>(std::cos(angle));
            float const sinAngle = static_cast<float>(std::sin(angle));
            std::complex<float> const phase{cosAngle, sinAngle};

            tileBuffer[localIdx] = localValue * phase;
        }
    }

    void printSpectrumPreview(int worldRank, std::vector<std::complex<float>> const& spectrum, std::size_t previewBins)
    {
        if(worldRank != 0 || spectrum.empty())
            return;

        std::size_t const bins = std::min(previewBins, spectrum.size());
        auto const originalFlags = std::cout.flags();
        auto const originalPrecision = std::cout.precision();
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "First " << bins << " frequency bins (real, imag, magnitude):\n";
        for(std::size_t k = 0; k < bins; ++k)
        {
            float magnitude = std::abs(spectrum[k]);
            std::cout << "  k=" << std::setw(4) << k << " : " << std::setw(12) << spectrum[k].real() << "  "
                      << std::setw(12) << spectrum[k].imag() << "i  |X|=" << magnitude << '\n';
        }
        auto maxIt = std::max_element(
            spectrum.begin(),
            spectrum.end(),
            [](std::complex<float> const& lhs, std::complex<float> const& rhs)
            { return std::abs(lhs) < std::abs(rhs); });
        if(maxIt != spectrum.end())
        {
            std::size_t const idx = static_cast<std::size_t>(std::distance(spectrum.begin(), maxIt));
            std::cout << "Dominant bin: k=" << idx << " |X|=" << std::abs(*maxIt) << '\n';
        }
        std::cout.flags(originalFlags);
        std::cout.precision(originalPrecision);
    }
} // namespace collectiveFft::detail
