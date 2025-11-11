/* cuFFT Provider Implementation
 * Bridges Alpaka tensor FFT requests to NVIDIA's cuFFT library.
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once

#include <alpaka/onHost/interface.hpp>
#include <alpaka/tensor/providers/ProviderInterface.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <complex>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

#ifdef ALPAKA_HAS_CUFFT
#    include <cuda_runtime.h>
#    include <cufft.h>
#endif

namespace alpaka::tensor
{
    class CuFFTProvider : public IOpProvider
    {
    private:
#ifdef ALPAKA_HAS_CUFFT
        struct PlanConfig
        {
            std::array<int, 3> lengths{1, 1, 1};
            int rank = 1;
            int batch = 1;
            ops::FftNumericType numeric = ops::FftNumericType::ComplexFloat32;
            bool inPlace = false;

            bool operator==(PlanConfig const&) const = default;
        };

        mutable cufftHandle plan_{0};
        mutable bool planInitialized_{false};
        mutable PlanConfig cachedConfig_{};
#endif

    public:
        std::string getBackendName() const override
        {
#ifdef ALPAKA_HAS_CUFFT
            return "cuFFT (CUDA)";
#else
            return "cuFFT (unavailable)";
#endif
        }

        bool supportsOperation(OpType op) const override
        {
#ifdef ALPAKA_HAS_CUFFT
            return op == OpType::FFT;
#else
            (void) op;
            return false;
#endif
        }

        bool isActive() const override
        {
#ifdef ALPAKA_HAS_CUFFT
            if(!supportsOperation(OpType::FFT))
                return false;

            if(char const* disable = std::getenv("ALPAKA_DISABLE_CUFFT"))
            {
                std::string value(disable);
                std::transform(
                    value.begin(),
                    value.end(),
                    value.begin(),
                    [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
                if(value == "1" || value == "true" || value == "on")
                    return false;
            }
            return true;
#else
            return false;
#endif
        }

        ~CuFFTProvider() override
        {
#ifdef ALPAKA_HAS_CUFFT
            destroyPlan();
#endif
        }

        template<typename ComplexT, typename Exec, typename Device, typename Queue>
        void fft(
            Exec const& exec,
            Device const& device,
            Queue& queue,
            tensor::Tensor1D<ComplexT, Device>& input,
            tensor::Tensor1D<ComplexT, Device>& output,
            ops::FftParams const& params) const
        {
#ifdef ALPAKA_HAS_CUFFT
            static_assert(
                std::is_same_v<std::decay_t<Exec>, alpaka::exec::GpuCuda>,
                "CuFFTProvider only supports CUDA executors");
            static_assert(
                std::is_same_v<ComplexT, std::complex<float>> || std::is_same_v<ComplexT, std::complex<double>>,
                "CuFFTProvider supports complex float and complex double tensors only");

            if(!isActive())
            {
                throw std::runtime_error("cuFFT provider is not active");
            }

            auto const numeric = ops::detail::deduceNumericType<ComplexT>();
            bool const alias = &input == &output;
            PlanConfig config = makePlanConfig(params, numeric, alias || params.inPlace);
            auto const streamHandle = static_cast<cudaStream_t>(alpaka::onHost::getNativeHandle(queue));
            ensurePlan(config, streamHandle);

            input.ensureOnDevice(device, queue);
            output.ensureOnDevice(device, queue);

            auto& inputBuffer = input.deviceBuffer(device, queue);
            auto& outputBuffer = output.deviceBuffer(device, queue);
            auto* inputPtr = reinterpret_cast<void*>(alpaka::onHost::data(inputBuffer));
            auto* outputPtr = reinterpret_cast<void*>(alpaka::onHost::data(outputBuffer));

            int const direction = params.direction == ops::FftDirection::Forward ? CUFFT_FORWARD : CUFFT_INVERSE;
            cufftResult execStatus = CUFFT_SUCCESS;
            switch(numeric)
            {
            case ops::FftNumericType::ComplexFloat32:
                {
                    auto* in = static_cast<cufftComplex*>(inputPtr);
                    auto* out = static_cast<cufftComplex*>(alias ? inputPtr : outputPtr);
                    execStatus = cufftExecC2C(plan_, in, out, direction);
                    break;
                }
            case ops::FftNumericType::ComplexFloat64:
                {
                    auto* in = static_cast<cufftDoubleComplex*>(inputPtr);
                    auto* out = static_cast<cufftDoubleComplex*>(alias ? inputPtr : outputPtr);
                    execStatus = cufftExecZ2Z(plan_, in, out, direction);
                    break;
                }
            }

            if(execStatus != CUFFT_SUCCESS)
            {
                throw std::runtime_error("cuFFT execution failed with status " + std::to_string(execStatus));
            }

            output.markDeviceModified(device, queue);
            if(alias)
            {
                input.markDeviceModified(device, queue);
            }

            if(eagerHostEnabled())
            {
                output.toHost(device, queue);
                if(alias)
                    input.toHost(device, queue);
            }
#else
            (void) exec;
            (void) device;
            (void) queue;
            (void) input;
            (void) output;
            (void) params;
            throw std::runtime_error("cuFFT support not enabled in this build");
#endif
        }

    protected:
        using IOpProvider::fft_impl; // inherit default implementation

    private:
#ifdef ALPAKA_HAS_CUFFT
        static cufftType toCufftType(ops::FftNumericType numeric)
        {
            switch(numeric)
            {
            case ops::FftNumericType::ComplexFloat32:
                return CUFFT_C2C;
            case ops::FftNumericType::ComplexFloat64:
                return CUFFT_Z2Z;
            }
            return CUFFT_C2C;
        }

        static PlanConfig makePlanConfig(ops::FftParams const& params, ops::FftNumericType numeric, bool inPlace)
        {
            PlanConfig config{};
            config.numeric = numeric;
            config.inPlace = inPlace;
            config.rank = static_cast<int>(params.rank == 0U ? 1U : std::min<unsigned int>(params.rank, 3U));
            for(int i = 0; i < config.rank; ++i)
            {
                if(params.lengths[i] > static_cast<std::size_t>(std::numeric_limits<int>::max()))
                {
                    throw std::invalid_argument("FFT length exceeds cuFFT capacity");
                }
                config.lengths[static_cast<std::size_t>(i)] = static_cast<int>(params.lengths[i]);
            }
            for(int i = config.rank; i < 3; ++i)
            {
                config.lengths[static_cast<std::size_t>(i)] = 1;
            }
            if(params.batch == 0U || params.batch > static_cast<std::size_t>(std::numeric_limits<int>::max()))
            {
                throw std::invalid_argument("FFT batch size invalid for cuFFT");
            }
            config.batch = static_cast<int>(params.batch);
            return config;
        }

        static int computeDistance(PlanConfig const& config)
        {
            long long product = 1;
            for(int i = 0; i < config.rank; ++i)
            {
                product *= static_cast<long long>(config.lengths[static_cast<std::size_t>(i)]);
            }
            if(product <= 0 || product > static_cast<long long>(std::numeric_limits<int>::max()))
            {
                throw std::invalid_argument("FFT size exceeds cuFFT integer limits");
            }
            return static_cast<int>(product);
        }

        void ensurePlan(PlanConfig const& config, cudaStream_t stream) const
        {
            if(planInitialized_ && config == cachedConfig_)
            {
                setStream(stream);
                return;
            }

            destroyPlan();

            int const distance = computeDistance(config);
            std::array<int, 3> dims = config.lengths;
            cufftType const type = toCufftType(config.numeric);
            cufftResult const status = cufftPlanMany(
                &plan_,
                config.rank,
                dims.data(),
                nullptr,
                1,
                distance,
                nullptr,
                1,
                distance,
                type,
                config.batch);
            if(status != CUFFT_SUCCESS)
            {
                plan_ = 0;
                throw std::runtime_error("cufftPlanMany failed with status " + std::to_string(status));
            }

            setStream(stream);
            planInitialized_ = true;
            cachedConfig_ = config;
        }

        void setStream(cudaStream_t stream) const
        {
            if(planInitialized_ && plan_ != 0)
            {
                cufftSetStream(plan_, stream);
            }
        }

        void destroyPlan() const
        {
            if(planInitialized_ && plan_ != 0)
            {
                cufftDestroy(plan_);
            }
            plan_ = 0;
            planInitialized_ = false;
        }

        static bool eagerHostEnabled()
        {
            return std::getenv("ALPAKA_EAGER_HOST") != nullptr;
        }
#else
        static bool eagerHostEnabled()
        {
            return false;
        }
#endif
    };
} // namespace alpaka::tensor
