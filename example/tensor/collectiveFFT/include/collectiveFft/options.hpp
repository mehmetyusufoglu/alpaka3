#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace collectiveFft
{
    enum class DirectVerifyPrecision
    {
        Float32,
        Float64
    };

    struct CommandLineOptions
    {
        std::size_t signalLength = 16384;
        std::optional<std::string> signalFile{};
        std::size_t previewBins = 12;
        bool verifyDirect = true;
        std::optional<std::string> referenceFftFile{};
        float verifyAbsTolerance = 1.0e-4f;
        float verifyRelTolerance = 1.0e-3f;
        DirectVerifyPrecision verifyPrecision = DirectVerifyPrecision::Float64;
        std::vector<std::string> warnings{};
    };

    CommandLineOptions parseCommandLine(int argc, char** argv);
} // namespace collectiveFft
