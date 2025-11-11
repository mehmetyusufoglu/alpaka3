#pragma once

#include "collectiveFft/options.hpp"

#include <complex>
#include <cstddef>
#include <string>
#include <vector>

namespace collectiveFft
{
    bool loadSignalChunk(
        CommandLineOptions const& options,
        std::size_t samplesPerRank,
        std::size_t totalSamples,
        int worldRank,
        std::size_t worldSize,
        std::vector<std::complex<float>>& chunk,
        std::string& errorMessage);

    bool loadFullSignalSequence(
        CommandLineOptions const& options,
        std::size_t totalSamples,
        std::vector<std::complex<float>>& samples,
        std::string& errorMessage);

    bool loadReferenceSpectrumFromFile(
        std::string const& path,
        std::size_t totalSamples,
        std::vector<std::complex<float>>& spectrum,
        std::string& errorMessage);
} // namespace collectiveFft
