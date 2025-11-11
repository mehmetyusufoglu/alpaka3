/* Unified Provider Interface & Concepts
 * Provides: OpType, OpStatus, IOpProvider base, C++20 concepts for capability checking.
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once

#include <alpaka/tensor/core/TensorTypes.hpp>

#include <array>
#include <complex>
#include <concepts>
#include <cstddef>
#include <string>
#include <type_traits>

namespace alpaka::tensor
{
    enum class OpType
    {
        Conv2D,
        GEMM,
        BatchNorm,
        Pooling,
        Activation,
        FFT
    };

    enum class OpStatus
    {
        Success, // Operation completed successfully
        Unsupported, // Operation not supported by this provider
        Error // Operation failed due to error
    };

    // Forward declaration of operation parameter types
    namespace ops
    {
        struct Conv2DParams;
        struct Pool2DParams;

        enum class FftTransformType
        {
            ComplexToComplex,
            RealToComplex,
            ComplexToReal
        };

        enum class FftDirection
        {
            Forward,
            Inverse
        };

        enum class FftNumericType
        {
            ComplexFloat32,
            ComplexFloat64
        };

        struct FftParams
        {
            std::array<std::size_t, 3> lengths{1, 1, 1};
            unsigned int rank = 1;
            std::size_t batch = 1;
            FftTransformType transformType = FftTransformType::ComplexToComplex;
            FftDirection direction = FftDirection::Forward;
            bool inPlace = false;
        };

        namespace detail
        {
            template<typename ComplexT>
            consteval FftNumericType deduceNumericType()
            {
                if constexpr(std::is_same_v<ComplexT, std::complex<float>>)
                {
                    return FftNumericType::ComplexFloat32;
                }
                else if constexpr(std::is_same_v<ComplexT, std::complex<double>>)
                {
                    return FftNumericType::ComplexFloat64;
                }
                else
                {
                    static_assert(sizeof(ComplexT) == 0, "Unsupported complex type for FFT provider");
                }
            }
        } // namespace detail
    } // namespace ops

    // Polymorphic base (runtime optional). Type-erased virtual layer kept minimal.
    class IOpProvider
    {
    public:
        virtual ~IOpProvider() = default;
        virtual std::string getBackendName() const = 0;
        virtual bool supportsOperation(OpType) const = 0;
        virtual bool isActive() const = 0;

        // ------------------------------------------------------------------
        // Type-safe front-end wrappers that forward to the protected
        // type-erased virtuals. These mirror the signatures used previously
        // (conv2d_status, gemm_status, batchnorm_status) so that existing
        // coordination code (CleanTensorOpContext, ops helpers) does not need
        // to change after the refactor that introduced this unified header.
        // ------------------------------------------------------------------

        // Convolution
        template<typename T, typename Exec, typename Device, typename Queue>
        OpStatus conv2d_status(
            Exec const& exec,
            Device const& device,
            Queue& queue,
            tensor::Tensor4D<T, Device> const& input,
            tensor::Tensor4D<T, Device> const& weight,
            ops::Conv2DParams const& params,
            tensor::Tensor4D<T, Device>& out)
        {
            return conv2d_impl(
                static_cast<void const*>(&exec),
                static_cast<void const*>(&device),
                static_cast<void*>(&queue),
                static_cast<void const*>(&input),
                static_cast<void const*>(&weight),
                params,
                static_cast<void*>(&out));
        }

        // GEMM
        template<typename Exec, typename Device, typename Queue>
        OpStatus gemm_status(
            Exec const& exec,
            Device const& device,
            Queue& queue,
            std::size_t M,
            std::size_t N,
            std::size_t K,
            float alpha,
            tensor::Tensor1D<float, Device>& A,
            tensor::Tensor1D<float, Device>& B,
            float beta,
            tensor::Tensor1D<float, Device>& C)
        {
            return gemm_impl(
                static_cast<void const*>(&exec),
                static_cast<void const*>(&device),
                static_cast<void*>(&queue),
                M,
                N,
                K,
                alpha,
                static_cast<void const*>(&A),
                static_cast<void const*>(&B),
                beta,
                static_cast<void*>(&C));
        }

        // BatchNorm
        template<typename T, typename Exec, typename Device, typename Queue>
        OpStatus batchnorm_status(
            Exec const& exec,
            Device const& device,
            Queue& queue,
            tensor::Tensor4D<T, Device> const& input,
            tensor::Tensor1D<T, Device> const& mean,
            tensor::Tensor1D<T, Device> const& variance,
            tensor::Tensor1D<T, Device> const& gamma,
            tensor::Tensor1D<T, Device> const& beta,
            T epsilon,
            tensor::Tensor4D<T, Device>& out)
        {
            return batchnorm_impl(
                static_cast<void const*>(&exec),
                static_cast<void const*>(&device),
                static_cast<void*>(&queue),
                static_cast<void const*>(&input),
                static_cast<void const*>(&mean),
                static_cast<void const*>(&variance),
                static_cast<void const*>(&gamma),
                static_cast<void const*>(&beta),
                static_cast<float>(epsilon),
                static_cast<void*>(&out));
        }

        template<typename ComplexT, typename Exec, typename Device, typename Queue>
        OpStatus fft_status(
            Exec const& exec,
            Device const& device,
            Queue& queue,
            tensor::Tensor1D<ComplexT, Device>& input,
            tensor::Tensor1D<ComplexT, Device>& output,
            ops::FftParams const& params)
        {
            return fft_impl(
                static_cast<void const*>(&exec),
                static_cast<void const*>(&device),
                static_cast<void*>(&queue),
                static_cast<void*>(&input),
                static_cast<void*>(&output),
                params,
                ops::detail::deduceNumericType<ComplexT>());
        }

    protected:
        virtual OpStatus conv2d_impl(
            void const*,
            void const*,
            void*,
            void const*,
            void const*,
            ops::Conv2DParams const&,
            void*)
        {
            return OpStatus::Unsupported;
        }

        virtual OpStatus gemm_impl(
            void const*,
            void const*,
            void*,
            std::size_t,
            std::size_t,
            std::size_t,
            float,
            void const*,
            void const*,
            float,
            void*)
        {
            return OpStatus::Unsupported;
        }

        virtual OpStatus batchnorm_impl(
            void const*,
            void const*,
            void*,
            void const*,
            void const*,
            void const*,
            void const*,
            void const*,
            float,
            void*)
        {
            return OpStatus::Unsupported;
        }

        virtual OpStatus fft_impl(
            void const*,
            void const*,
            void*,
            void*,
            void*,
            ops::FftParams const&,
            ops::FftNumericType)
        {
            return OpStatus::Unsupported;
        }
    };

    // Concepts (lightweight detection of typed APIs)
    template<typename P>
    concept ProviderBase = requires(P p) {
        { p.getBackendName() } -> std::convertible_to<std::string>;
        { p.isActive() } -> std::convertible_to<bool>;
        { p.supportsOperation(OpType::Conv2D) } -> std::convertible_to<bool>;
        { p.supportsOperation(OpType::FFT) } -> std::convertible_to<bool>;
        // TODO(ROCM): add optional compile-time query hooks for HIP feature flags if needed
    };

    template<typename P, typename Exec, typename Device, typename Queue, typename T>
    concept Conv2DProvider = ProviderBase<P>
                             && requires(
                                 P const& p,
                                 Exec const& e,
                                 Device const& d,
                                 Queue& q,
                                 Tensor4D<T, Device> const& x,
                                 Tensor4D<T, Device> const& w,
                                 ops::Conv2DParams const& params) {
                                    { p.conv2d(e, d, q, x, w, params) };
                                };

    template<typename P, typename Exec, typename Device, typename Queue>
    concept GemmProvider = ProviderBase<P>
                           && requires(
                               P const& p,
                               Exec const& e,
                               Device const& d,
                               Queue& q,
                               std::size_t M,
                               std::size_t N,
                               std::size_t K,
                               float alpha,
                               Tensor1D<float, Device>& A,
                               Tensor1D<float, Device>& B,
                               float beta,
                               Tensor1D<float, Device>& C) {
                                  { p.gemm(e, d, q, M, N, K, alpha, A, B, beta, C) };
                              };

    template<typename P, typename Exec, typename Device, typename Queue, typename ComplexT>
    concept FftProvider = ProviderBase<P>
                          && requires(
                              P const& p,
                              Exec const& e,
                              Device const& d,
                              Queue& q,
                              tensor::Tensor1D<ComplexT, Device>& input,
                              tensor::Tensor1D<ComplexT, Device>& output,
                              ops::FftParams const& params) {
                                 { p.fft(e, d, q, input, output, params) };
                             };
} // namespace alpaka::tensor
