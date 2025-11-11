#include "collectiveFft/options.hpp"

#include <cmath>
#include <cstdlib>
#include <string>
#include <string_view>
#include <utility>

namespace collectiveFft
{
    CommandLineOptions parseCommandLine(int argc, char** argv)
    {
        CommandLineOptions options{};
        constexpr std::string_view lengthPrefix{"--signal-length="};
        constexpr std::string_view filePrefix{"--signal-file="};
        constexpr std::string_view previewPrefix{"--preview-bins="};
        constexpr std::string_view referencePrefix{"--reference-fft="};
        constexpr std::string_view absTolPrefix{"--verify-abs="};
        constexpr std::string_view relTolPrefix{"--verify-rel="};
        constexpr std::string_view verifyFlag{"--verify-direct"};
        constexpr std::string_view skipVerifyFlag{"--skip-verify"};
        constexpr std::string_view precisionPrefix{"--verify-direct-precision="};
        constexpr std::string_view disableProviderFlag{"--disable-provider-fft"};
        constexpr std::string_view forceProviderFlag{"--force-provider-fft"};

        for(int i = 1; i < argc; ++i)
        {
            std::string_view arg(argv[i]);
            if(arg.rfind(lengthPrefix, 0) == 0)
            {
                std::string value(arg.substr(lengthPrefix.size()));
                char* end = nullptr;
                unsigned long long parsed = std::strtoull(value.c_str(), &end, 10);
                if(end == value.c_str() || parsed == 0ULL)
                {
                    options.warnings.emplace_back(
                        "Ignoring invalid --signal-length value '" + value + "'; keeping default.");
                }
                else
                {
                    options.signalLength = static_cast<std::size_t>(parsed);
                }
            }
            else if(arg.rfind(filePrefix, 0) == 0)
            {
                std::string path(arg.substr(filePrefix.size()));
                if(path.empty())
                {
                    options.warnings.emplace_back("Ignoring empty --signal-file argument.");
                }
                else
                {
                    options.signalFile = std::move(path);
                }
            }
            else if(arg.rfind(previewPrefix, 0) == 0)
            {
                std::string value(arg.substr(previewPrefix.size()));
                char* end = nullptr;
                unsigned long long parsed = std::strtoull(value.c_str(), &end, 10);
                if(end == value.c_str() || parsed == 0ULL)
                {
                    options.warnings.emplace_back(
                        "Ignoring invalid --preview-bins value '" + value + "'; keeping default.");
                }
                else
                {
                    options.previewBins = static_cast<std::size_t>(parsed);
                }
            }
            else if(arg == skipVerifyFlag)
            {
                options.verifyDirect = false;
            }
            else if(arg == verifyFlag)
            {
                options.verifyDirect = true;
            }
            else if(arg == disableProviderFlag)
            {
                setProviderFftEnabled(options, false);
                setProviderFftRequired(options, false);
            }
            else if(arg == forceProviderFlag)
            {
                setProviderFftEnabled(options, true);
                setProviderFftRequired(options, true);
            }
            else if(arg.rfind(referencePrefix, 0) == 0)
            {
                std::string path(arg.substr(referencePrefix.size()));
                if(path.empty())
                {
                    options.warnings.emplace_back("Ignoring empty --reference-fft argument.");
                }
                else
                {
                    options.referenceFftFile = std::move(path);
                }
            }
            else if(arg.rfind(absTolPrefix, 0) == 0)
            {
                std::string value(arg.substr(absTolPrefix.size()));
                char* end = nullptr;
                float parsed = std::strtof(value.c_str(), &end);
                if(end == value.c_str() || !std::isfinite(parsed) || parsed < 0.0f)
                {
                    options.warnings.emplace_back(
                        "Ignoring invalid --verify-abs value '" + value + "'; keeping default.");
                }
                else
                {
                    options.verifyAbsTolerance = parsed;
                }
            }
            else if(arg.rfind(relTolPrefix, 0) == 0)
            {
                std::string value(arg.substr(relTolPrefix.size()));
                char* end = nullptr;
                float parsed = std::strtof(value.c_str(), &end);
                if(end == value.c_str() || !std::isfinite(parsed) || parsed < 0.0f)
                {
                    options.warnings.emplace_back(
                        "Ignoring invalid --verify-rel value '" + value + "'; keeping default.");
                }
                else
                {
                    options.verifyRelTolerance = parsed;
                }
            }
            else if(arg.rfind(precisionPrefix, 0) == 0)
            {
                std::string value(arg.substr(precisionPrefix.size()));
                if(value == "float" || value == "Float" || value == "float32")
                {
                    options.verifyPrecision = DirectVerifyPrecision::Float32;
                }
                else if(value == "double" || value == "Double" || value == "float64")
                {
                    options.verifyPrecision = DirectVerifyPrecision::Float64;
                }
                else
                {
                    options.warnings.emplace_back(
                        "Ignoring unknown --verify-direct-precision value '" + value
                        + "'; expected 'float' or 'double'.");
                }
            }
        }

        if(options.signalLength == 0)
            options.signalLength = 1;
        if(options.previewBins == 0)
            options.previewBins = 1;
        return options;
    }
} // namespace collectiveFft
