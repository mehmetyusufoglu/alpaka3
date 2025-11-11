/* Centralized fallback tensor operations
 * Thin wrappers around existing generic kernels so providers can
 * reuse a single header when they need to execute the non-vendor path.
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once

#include <alpaka/tensor/ops/convolution/Conv2D.hpp>
#include <alpaka/tensor/ops/elementwise/ActivationOps.hpp>
#include <alpaka/tensor/ops/fft/NaiveFft.hpp>
#include <alpaka/tensor/ops/linear/Gemm.hpp>
#include <alpaka/tensor/ops/normalization/BatchNorm.hpp>
#include <alpaka/tensor/ops/pooling/Pooling.hpp>
#include <alpaka/tensor/ops/pooling/PoolingTypes.hpp>

#include <complex>
#include <cstddef>
#include <span>
#include <type_traits>
#include <vector>

namespace alpaka::tensor::ops::fallback
{
    template<typename T, typename Exec, typename Device, typename Queue>
    auto conv2d(
        Exec const& exec,
        Device const& device,
        Queue& queue,
        tensor::Tensor4D<T, Device> const& input,
        tensor::Tensor4D<T, Device> const& weight,
        ops::Conv2DParams const& params) -> tensor::Tensor4D<T, Device>
    {
        return ::alpaka::tensor::ops::conv2d<T>(exec, device, queue, input, weight, params);
    }

    template<typename Exec, typename Device, typename Queue>
    void gemm(
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
        ::alpaka::tensor::ops::gemm(exec, device, queue, 'N', 'N', M, N, K, alpha, A, B, beta, C);
    }

    template<typename ComplexT, typename Exec, typename Device, typename Queue>
    void fft(
        Exec const& exec,
        Device const& device,
        Queue& queue,
        tensor::Tensor1D<ComplexT, Device>& input,
        tensor::Tensor1D<ComplexT, Device>& output,
        ops::FftParams const& params)
    {
        static_assert(
            std::is_same_v<ComplexT, std::complex<float>> || std::is_same_v<ComplexT, std::complex<double>>,
            "Fallback FFT requires std::complex<float> or std::complex<double>");

        (void) exec;

        input.toHost(device, queue);
        std::span<ComplexT const> inputSpan{input.hostData(), input.size()};

        std::vector<ComplexT> hostOutput(inputSpan.size());
        std::span<ComplexT> outputSpan{hostOutput.data(), hostOutput.size()};

        ops::fft::naiveNdFft(inputSpan, outputSpan, params);

        if(&input == &output)
        {
            auto* hostData = input.hostData();
            std::copy(hostOutput.begin(), hostOutput.end(), hostData);
            input.markHostModified();
        }
        else
        {
            output.toHost(device, queue);
            auto* hostData = output.hostData();
            std::copy(hostOutput.begin(), hostOutput.end(), hostData);
            output.markHostModified();
        }
    }

    template<typename T, typename Exec, typename Device, typename Queue>
    auto batchnorm(
        Exec const& exec,
        Device const& device,
        Queue& queue,
        tensor::Tensor4D<T, Device> const& input,
        tensor::Tensor1D<T, Device> const& mean,
        tensor::Tensor1D<T, Device> const& variance,
        tensor::Tensor1D<T, Device> const& gamma,
        tensor::Tensor1D<T, Device> const& beta,
        T epsilon) -> tensor::Tensor4D<T, Device>
    {
        tensor::Tensor4D<T, Device> output(device, input.shape());
        auto& inRef = const_cast<tensor::Tensor4D<T, Device>&>(input);
        auto& meanRef = const_cast<tensor::Tensor1D<T, Device>&>(mean);
        auto& varRef = const_cast<tensor::Tensor1D<T, Device>&>(variance);
        auto& gammaRef = const_cast<tensor::Tensor1D<T, Device>&>(gamma);
        auto& betaRef = const_cast<tensor::Tensor1D<T, Device>&>(beta);
        ::alpaka::tensor::ops::batch_norm_inference(
            exec,
            device,
            queue,
            inRef,
            gammaRef,
            betaRef,
            meanRef,
            varRef,
            epsilon,
            output);
        return output;
    }

    template<typename T, typename Exec, typename Device, typename Queue>
    auto max_pool2d(
        Exec const& exec,
        Device const& device,
        Queue& queue,
        tensor::Tensor4D<T, Device>& input,
        ops::Pool2DParams const& params) -> tensor::Tensor4D<T, Device>
    {
        return ::alpaka::tensor::ops::max_pool2d<T>(exec, device, queue, input, params);
    }

    template<typename T, typename Exec, typename Device, typename Queue>
    auto avg_pool2d(
        Exec const& exec,
        Device const& device,
        Queue& queue,
        tensor::Tensor4D<T, Device>& input,
        ops::Pool2DParams const& params) -> tensor::Tensor4D<T, Device>
    {
        return ::alpaka::tensor::ops::avg_pool2d<T>(exec, device, queue, input, params);
    }

    template<typename T, std::size_t Rank, typename Exec, typename Device, typename Queue>
    void relu_inplace(Exec const& exec, Device const& device, Queue& queue, tensor::Tensor<T, Rank, Device>& t)
    {
        ::alpaka::tensor::ops::relu_inplace(exec, device, queue, t);
    }

    template<typename T, std::size_t Rank, typename Exec, typename Device, typename Queue>
    void gelu(Exec const& exec, Device const& device, Queue& queue, tensor::Tensor<T, Rank, Device>& t)
    {
        ::alpaka::tensor::ops::gelu<T>(exec, device, queue, t);
    }
} // namespace alpaka::tensor::ops::fallback
