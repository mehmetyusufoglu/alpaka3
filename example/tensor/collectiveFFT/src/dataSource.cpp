#include "collectiveFft/dataSource.hpp"

#include <cmath>
#include <fstream>
#include <numbers>

namespace collectiveFft
{
    namespace
    {
        float syntheticSample(std::size_t globalIndex, std::size_t totalLength)
        {
            constexpr float twoPi = 2.0f * std::numbers::pi_v<float>;
            float position
                = (totalLength > 0) ? static_cast<float>(globalIndex) / static_cast<float>(totalLength) : 0.0f;
            float base = std::sin(twoPi * 3.0f * position);
            float overtone = 0.35f * std::sin(twoPi * 17.0f * position + 0.6f);
            float envelope = 0.5f * std::cos(twoPi * 1.0f * position);
            return base + overtone + envelope;
        }
    } // namespace

    bool loadSignalChunk(
        CommandLineOptions const& options,
        std::size_t samplesPerRank,
        std::size_t totalSamples,
        int worldRank,
        std::size_t worldSize,
        std::vector<std::complex<float>>& chunk,
        std::string& errorMessage)
    {
        chunk.assign(samplesPerRank, std::complex<float>{0.0f, 0.0f});
        std::size_t const stride = worldSize;

        if(options.signalFile)
        {
            std::ifstream input(options.signalFile->c_str(), std::ios::binary);
            if(!input)
            {
                errorMessage = "Failed to open signal file '" + *options.signalFile + "'.";
                return false;
            }

            for(std::size_t m = 0; m < samplesPerRank; ++m)
            {
                std::size_t const globalIndex = stride * m + static_cast<std::size_t>(worldRank);
                if(globalIndex >= totalSamples)
                    break;

                std::streamoff const offset
                    = static_cast<std::streamoff>(globalIndex) * static_cast<std::streamoff>(sizeof(float));
                input.seekg(offset, std::ios::beg);
                if(!input.good())
                {
                    errorMessage = "seekg failed at byte offset " + std::to_string(offset);
                    return false;
                }

                float sample = 0.0f;
                input.read(reinterpret_cast<char*>(&sample), sizeof(float));
                if(!input)
                {
                    errorMessage = "Failed to read sample index " + std::to_string(globalIndex);
                    return false;
                }

                chunk[m] = std::complex<float>{sample, 0.0f};
            }
        }
        else
        {
            for(std::size_t m = 0; m < samplesPerRank; ++m)
            {
                std::size_t const globalIndex = stride * m + static_cast<std::size_t>(worldRank);
                if(globalIndex >= totalSamples)
                    break;
                chunk[m] = std::complex<float>{syntheticSample(globalIndex, totalSamples), 0.0f};
            }
        }

        return true;
    }

    bool loadFullSignalSequence(
        CommandLineOptions const& options,
        std::size_t totalSamples,
        std::vector<std::complex<float>>& samples,
        std::string& errorMessage)
    {
        samples.assign(totalSamples, std::complex<float>{0.0f, 0.0f});
        if(totalSamples == 0)
            return true;

        if(options.signalFile)
        {
            std::ifstream input(options.signalFile->c_str(), std::ios::binary);
            if(!input)
            {
                errorMessage = "Failed to open signal file '" + *options.signalFile + "' for verification.";
                return false;
            }

            input.seekg(0, std::ios::end);
            std::streamoff const fileBytes = input.tellg();
            if(fileBytes < 0)
            {
                errorMessage = "tellg failed while sizing signal file.";
                return false;
            }
            std::size_t const expectedBytes = totalSamples * sizeof(float);
            if(static_cast<std::size_t>(fileBytes) < expectedBytes)
            {
                errorMessage = "Signal file shorter than required samples (expected at least "
                               + std::to_string(expectedBytes) + " bytes).";
                return false;
            }
            input.seekg(0, std::ios::beg);

            for(std::size_t idx = 0; idx < totalSamples; ++idx)
            {
                float value = 0.0f;
                input.read(reinterpret_cast<char*>(&value), sizeof(float));
                if(!input)
                {
                    errorMessage = "Failed to read sample index " + std::to_string(idx) + " during verification.";
                    return false;
                }
                samples[idx] = std::complex<float>{value, 0.0f};
            }
        }
        else
        {
            for(std::size_t idx = 0; idx < totalSamples; ++idx)
            {
                samples[idx] = std::complex<float>{syntheticSample(idx, totalSamples), 0.0f};
            }
        }

        return true;
    }

    bool loadReferenceSpectrumFromFile(
        std::string const& path,
        std::size_t totalSamples,
        std::vector<std::complex<float>>& spectrum,
        std::string& errorMessage)
    {
        spectrum.assign(totalSamples, std::complex<float>{0.0f, 0.0f});
        if(totalSamples == 0)
            return true;

        std::ifstream input(path.c_str(), std::ios::binary);
        if(!input)
        {
            errorMessage = "Failed to open reference FFT file '" + path + "'.";
            return false;
        }

        input.seekg(0, std::ios::end);
        std::streamoff const fileBytes = input.tellg();
        if(fileBytes < 0)
        {
            errorMessage = "tellg failed while sizing reference FFT file.";
            return false;
        }
        std::size_t const expectedBytes = totalSamples * sizeof(float) * 2U;
        if(static_cast<std::size_t>(fileBytes) < expectedBytes)
        {
            errorMessage = "Reference FFT file shorter than required spectrum length (expected at least "
                           + std::to_string(expectedBytes) + " bytes).";
            return false;
        }
        input.seekg(0, std::ios::beg);

        for(std::size_t idx = 0; idx < totalSamples; ++idx)
        {
            float real = 0.0f;
            float imag = 0.0f;
            input.read(reinterpret_cast<char*>(&real), sizeof(float));
            input.read(reinterpret_cast<char*>(&imag), sizeof(float));
            if(!input)
            {
                errorMessage = "Failed to read complex spectrum value at index " + std::to_string(idx) + '.';
                return false;
            }
            spectrum[idx] = std::complex<float>{real, imag};
        }

        return true;
    }
} // namespace collectiveFft
