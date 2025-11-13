#pragma once

#include "collectiveFft/bootstrap.hpp"

#include <complex>
#include <cstddef>
#include <optional>
#include <span>
#include <vector>

namespace collectiveFft::detail
{
    [[nodiscard]] std::optional<int> discoverLocalRank(MultiProcessBootstrap const& bootstrap);

    void buildContributionTile(
        std::span<std::complex<float> const> localSpectrum,
        int worldRank,
        std::size_t worldSize,
        std::size_t totalSamples,
        std::size_t tileOffset,
        std::span<std::complex<float>> tileBuffer);

    void printSpectrumPreview(
        int worldRank,
        std::vector<std::complex<float>> const& spectrum,
        std::size_t previewBins);
} // namespace collectiveFft::detail
