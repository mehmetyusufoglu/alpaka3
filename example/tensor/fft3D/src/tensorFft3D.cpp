/**
 * Example invocation:
 *   ./example/tensor/fft3D/tensorFft3D --nx=256 --ny=256 --nz=256 --batch=4 --force-provider-fft --verbose
 *
 * This sample synthesizes one or more complex-valued 3D volumes, then uses Alpaka's CleanTensorOpContext to
 * launch a batched complex-to-complex FFT via the vendor backend (cuFFT when available). The context owns the
 * concrete cuFFT plans, so the user-facing code stays backend-agnostic while still benefiting from vendor tuning.
 *
 * Batch size determines how many independent volumes are transformed in one call: all batches are stored
 * consecutively in device tensors and processed together to amortize plan setup. With `batch=1`, the largest cube
 * accepted by cuFFT today is 512^3 (~134M complex samples, ~1.1 GiB per buffer). Higher batch counts multiply both
 * the data set size and the device memory requirement accordingly, so keep batch small when exploring the limits.
 */
#include <alpaka/alpaka.hpp>
#include <alpaka/onHost/example/executors.hpp>
#include <alpaka/onHost/executeForEach.hpp>
#include <alpaka/onHost/interface.hpp>
#include <alpaka/tensor/providers/CleanTensorOpContext.hpp>
#include <alpaka/tensor/providers/EnabledVendorLibs.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <numbers>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace tt = alpaka::tensor;

namespace
{
    struct Fft3dOptions
    {
        std::size_t nx = 16;
        std::size_t ny = 16;
        std::size_t nz = 16;
        std::size_t batch = 1;
        bool verifyInverse = true;
        bool verbose = false;
        bool requireProvider = false;
    };

    std::optional<std::size_t> parsePositiveSize(std::string_view text)
    {
        try
        {
            auto value = static_cast<std::size_t>(std::stoull(std::string(text)));
            if(value == 0)
                return std::nullopt;
            return value;
        }
        catch(...)
        {
            return std::nullopt;
        }
    }

    Fft3dOptions parseCommandLine(int argc, char** argv)
    {
        Fft3dOptions options{};
        for(int i = 1; i < argc; ++i)
        {
            std::string_view arg(argv[i]);
            if(arg == "--no-verify")
            {
                options.verifyInverse = false;
            }
            else if(arg == "--verbose")
            {
                options.verbose = true;
            }
            else if(arg == "--force-provider-fft")
            {
                options.requireProvider = true;
            }
            else if(arg.rfind("--size=", 0) == 0)
            {
                if(auto value = parsePositiveSize(arg.substr(7)))
                {
                    options.nx = options.ny = options.nz = *value;
                }
                else
                {
                    std::cerr << "Ignoring invalid --size argument '" << arg.substr(7) << "'.\n";
                }
            }
            else if(arg.rfind("--nx=", 0) == 0)
            {
                if(auto value = parsePositiveSize(arg.substr(5)))
                    options.nx = *value;
                else
                    std::cerr << "Ignoring invalid --nx argument.\n";
            }
            else if(arg.rfind("--ny=", 0) == 0)
            {
                if(auto value = parsePositiveSize(arg.substr(5)))
                    options.ny = *value;
                else
                    std::cerr << "Ignoring invalid --ny argument.\n";
            }
            else if(arg.rfind("--nz=", 0) == 0)
            {
                if(auto value = parsePositiveSize(arg.substr(5)))
                    options.nz = *value;
                else
                    std::cerr << "Ignoring invalid --nz argument.\n";
            }
            else if(arg.rfind("--batch=", 0) == 0)
            {
                if(auto value = parsePositiveSize(arg.substr(8)))
                    options.batch = *value;
                else
                    std::cerr << "Ignoring invalid --batch argument.\n";
            }
            else
            {
                std::cerr << "Unrecognized option '" << arg << "'.\n";
            }
        }
        return options;
    }

    std::vector<std::complex<float>> generateSignal(Fft3dOptions const& options)
    {
        std::size_t const elementsPerTransform = options.nx * options.ny * options.nz;
        std::size_t const totalElements = elementsPerTransform * options.batch;
        std::vector<std::complex<float>> signal(totalElements, std::complex<float>{0.0f, 0.0f});

        float const twoPi = 2.0f * std::numbers::pi_v<float>;
        std::size_t index = 0;
        for(std::size_t b = 0; b < options.batch; ++b)
        {
            for(std::size_t z = 0; z < options.nz; ++z)
            {
                for(std::size_t y = 0; y < options.ny; ++y)
                {
                    for(std::size_t x = 0; x < options.nx; ++x)
                    {
                        float const phaseX = twoPi * static_cast<float>(x) / static_cast<float>(options.nx);
                        float const phaseY = twoPi * static_cast<float>(y) / static_cast<float>(options.ny);
                        float const phaseZ = twoPi * static_cast<float>(z) / static_cast<float>(options.nz);
                        float const batchPhase = 0.25f * static_cast<float>(b);
                        float magnitude = std::sin(phaseX + batchPhase) + std::cos(phaseY - batchPhase);
                        float imag = std::sin(phaseZ + phaseX * 0.5f) - std::cos(phaseY);
                        signal[index++] = std::complex<float>{magnitude, imag};
                    }
                }
            }
        }
        return signal;
    }

    void printSpectrumPreview(
        Fft3dOptions const& options,
        std::complex<float> const* spectrum,
        std::size_t spectrumElements)
    {
        if(!options.verbose || spectrumElements == 0)
            return;

        std::cout << "First 6 frequency bins (complex values):\n";
        std::size_t const preview = std::min<std::size_t>(6, spectrumElements);
        for(std::size_t idx = 0; idx < preview; ++idx)
        {
            std::cout << "  bin[" << idx << "] = " << spectrum[idx] << '\n';
        }
    }

    template<typename Tag>
    int runFft3d(Tag const& tag, Fft3dOptions const& options)
    {
        auto const& deviceSpec = tag[alpaka::object::deviceSpec];
        auto const& exec = tag[alpaka::object::exec];
        using Exec = std::decay_t<decltype(exec)>;

        if constexpr(!std::is_same_v<Exec, alpaka::exec::GpuCuda>)
        {
            std::cout << "Skipping backend " << deviceSpec.getApi().getName() << " (cuFFT required).\n";
            return 0;
        }
        else
        {
            auto selector = alpaka::onHost::makeDeviceSelector(deviceSpec);
            if(!selector.isAvailable())
            {
                std::cout << "No device available for " << deviceSpec.getApi().getName() << ".\n";
                return 0;
            }

            auto device = selector.makeDevice(0);
            auto queue = device.makeQueue();
            using Device = decltype(device);

            // CleanTensorOpContext owns the vendor libraries (cuFFT, etc.) and hands us an abstract FFT entrypoint.
            auto context = tt::createCleanTensorOpContext(exec, device, queue);
            bool const providerAvailable
                = tt::EnabledVendorLibs::hasCUFFT && context.supportsOperation(tt::OpType::FFT);
            if(options.requireProvider && !providerAvailable)
            {
                std::cerr << "cuFFT provider required (--force-provider-fft) but not available." << '\n';
                return 1;
            }

            // TODO(real FFT): once CleanTensorOpContext exposes R2C/C2R providers, add the real-signal path back.

            if(providerAvailable)
            {
                std::cout << "cuFFT being used for 3D FFT operations." << '\n';
            }
            else
            {
                std::cout << "Warning: cuFFT provider unavailable; naive fallback will execute on host." << '\n';
            }

            std::cout << "\n=== tensorFft3D ===\n";
            std::cout << "Device: " << deviceSpec.getApi().getName() << "\n";
            std::cout << "Grid: " << options.nx << " x " << options.ny << " x " << options.nz
                      << ", batch=" << options.batch << '\n';

            std::size_t const elementsPerTransform = options.nx * options.ny * options.nz;
            if(elementsPerTransform == 0 || options.batch == 0)
            {
                std::cerr << "Invalid dimensions specified." << '\n';
                return 1;
            }

            std::size_t const totalElements = elementsPerTransform * options.batch;
            // create one complex volume per batch entry; batches live back-to-back in the tensors
            auto hostSignal = generateSignal(options);

            tt::Tensor1D<std::complex<float>, Device> deviceInput(device, {totalElements}, "fft3d-input");
            tt::Tensor1D<std::complex<float>, Device> deviceOutput(device, {totalElements}, "fft3d-output");

            auto* hostInputPtr = deviceInput.hostData();
            std::copy(hostSignal.begin(), hostSignal.end(), hostInputPtr);
            deviceInput.markHostModified();

            tt::ops::FftParams forwardParams{};
            forwardParams.rank = 3;
            forwardParams.lengths = {options.nx, options.ny, options.nz};
            forwardParams.batch = options.batch;
            // cuFFT sees batch as the number of independent volumes transformed in one call
            forwardParams.transformType = tt::ops::FftTransformType::ComplexToComplex;
            forwardParams.direction = tt::ops::FftDirection::Forward;
            forwardParams.inPlace = false;

            // context.fft dispatches into cuFFT (when available) with the above batched plan metadata
            context.fft(deviceInput, deviceOutput, forwardParams);
            deviceOutput.toHost(device, queue);
            alpaka::onHost::wait(queue);

            printSpectrumPreview(options, deviceOutput.hostData(), totalElements);

            if(options.verifyInverse)
            {
                tt::ops::FftParams inverseParams = forwardParams;
                inverseParams.direction = tt::ops::FftDirection::Inverse;
                context.fft(deviceOutput, deviceInput, inverseParams);

                deviceInput.toHost(device, queue);
                alpaka::onHost::wait(queue);
                auto* recoveredPtr = deviceInput.hostData();

                double maxAbsError = 0.0;
                double maxRelError = 0.0;
                float const normalization = static_cast<float>(elementsPerTransform);

                for(std::size_t idx = 0; idx < totalElements; ++idx)
                {
                    std::complex<float> recovered = recoveredPtr[idx] / normalization;
                    std::complex<float> reference = hostSignal[idx];
                    std::complex<float> diff = recovered - reference;
                    double const absErr = std::abs(diff);
                    maxAbsError = std::max(maxAbsError, absErr);
                    double denom = std::max(1e-6, static_cast<double>(std::abs(reference)));
                    maxRelError = std::max(maxRelError, absErr / denom);
                }

                std::cout << "Inverse FFT max abs error: " << maxAbsError << ", max rel error: " << maxRelError
                          << '\n';
            }
            else
            {
                std::cout << "Inverse FFT verification skipped (--no-verify)." << '\n';
            }

            std::cout << "tensorFft3D complete.\n";
            return 0;
        }
    }
} // namespace

int main(int argc, char** argv)
{
    auto options = parseCommandLine(argc, argv);

    auto const result = alpaka::onHost::executeForEachIfHasDevice(
        [&options](auto const& tag) { return runFft3d(tag, options); },
        alpaka::onHost::allBackends(alpaka::onHost::enabledApis, alpaka::onHost::example::enabledExecutors));

    return result;
}
