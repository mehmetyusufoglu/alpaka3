#pragma once

#include <concepts>
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
        bool enableProviderFft = true;
        bool requireProviderFft = false;
        std::vector<std::string> warnings{};
    };

    CommandLineOptions parseCommandLine(int argc, char** argv);

    namespace detail
    {
        template<typename Options>
        concept HasEnableProviderFftMember = requires(Options& opts) { opts.enableProviderFft = true; };

        template<typename Options>
        concept HasRequireProviderFftMember = requires(Options& opts) { opts.requireProviderFft = true; };

        template<typename Options>
        concept HasWarningsVector = requires(Options& opts) { opts.warnings.emplace_back(std::string{}); };
    } // namespace detail

    inline bool providerFftEnabled(CommandLineOptions const& options)
    {
        if constexpr(detail::HasEnableProviderFftMember<CommandLineOptions>)
        {
            return options.enableProviderFft;
        }
        (void) options;
        return true;
    }

    inline bool providerFftRequired(CommandLineOptions const& options)
    {
        if constexpr(detail::HasRequireProviderFftMember<CommandLineOptions>)
        {
            return options.requireProviderFft;
        }
        (void) options;
        return false;
    }

    inline void setProviderFftEnabled(CommandLineOptions& options, bool enabled)
    {
        if constexpr(detail::HasEnableProviderFftMember<CommandLineOptions>)
        {
            options.enableProviderFft = enabled;
        }
        else
        {
            (void) enabled;
            (void) options;
            if constexpr(detail::HasWarningsVector<CommandLineOptions>)
            {
                if(!enabled)
                {
                    options.warnings.emplace_back(
                        "--disable-provider-fft ignored (cuFFT toggles unavailable in this build).");
                }
            }
        }
    }

    inline void setProviderFftRequired(CommandLineOptions& options, bool required)
    {
        if constexpr(detail::HasRequireProviderFftMember<CommandLineOptions>)
        {
            options.requireProviderFft = required;
        }
        else
        {
            (void) required;
            (void) options;
            if constexpr(detail::HasWarningsVector<CommandLineOptions>)
            {
                if(required)
                {
                    options.warnings.emplace_back(
                        "--force-provider-fft ignored (cuFFT toggles unavailable in this build).");
                }
            }
        }
    }
} // namespace collectiveFft
