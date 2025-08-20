/* Copyright 2025 Alpaka Tensor Library Contributors
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <alpaka/tensor/TensorGPUFixed.hpp>
#include <alpaka/alpaka.hpp>

namespace alpaka
{
    namespace tensor
    {
        namespace ops
        {
            // Elementwise addition kernel (following tutorial patterns)
            template<typename TAcc>
            class ElementwiseAddKernel {
            public:
                template<typename TBufA, typename TBufB, typename TBufR>
                ALPAKA_FN_ACC void operator()(TAcc const& acc, TBufA a, TBufB b, TBufR result, ::std::size_t n) const {
                    for(auto [index] : alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{n}))
                    {
                        result[index] = a[index] + b[index];
                    }
                }
            };

            // Elementwise ReLU kernel (activation function)
            template<typename TAcc>
            class ElementwiseReluKernel {
            public:
                template<typename TBufIn, typename TBufOut>
                ALPAKA_FN_ACC void operator()(TAcc const& acc, TBufIn in, TBufOut out, ::std::size_t n) const {
                    for(auto [index] : alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{n}))
                    {
                        auto v = in[index];
                        out[index] = v > decltype(v){} ? v : decltype(v){};
                    }
                }
            };

            // High-level addition operation (simplified - CPU fallback for now)
            template<typename T, ::std::size_t Rank>
            Tensor<T, Rank> add(const Tensor<T, Rank>& a, const Tensor<T, Rank>& b) {
                
                // Verify compatibility
                assert(a.isShapeCompatible(b) && "Tensors must have same shape");
                assert(a.isDeviceCompatible(b) && "Tensors must be on same device");
                
                // Create output tensor
                Tensor<T, Rank> result(a.shape(), a.dtype(), a.layout(), "add_result");
                
                // Simple CPU implementation for now
                T* data_a = const_cast<T*>(a.hostData());
                T* data_b = const_cast<T*>(b.hostData());
                T* data_result = result.hostData();
                
                ::std::size_t size = a.size();
                for(::std::size_t i = 0; i < size; ++i) {
                    data_result[i] = data_a[i] + data_b[i];
                }
                
                return result;
            }

            // Device-dispatch overload (explicit exec & queue)
            template<typename T, ::std::size_t Rank, typename Exec, typename Device, typename Queue>
            Tensor<T, Rank> add(Exec const& exec, Device const& device, Queue& queue, Tensor<T, Rank>& a, Tensor<T, Rank>& b) {
                assert(a.isShapeCompatible(b));
                // Ensure device buffers and upload if dirty
                a.ensureOnDevice(device, queue);
                b.ensureOnDevice(device, queue);
                Tensor<T, Rank> result(a.shape(), a.dtype(), a.layout(), "add_result");
                result.allocateDevice(device);
                // Kernel launch configuration (simple 1D)
                auto n = a.size();
                unsigned threadsPerBlock = 256u;
                unsigned blocks = static_cast<unsigned>((n + threadsPerBlock - 1) / threadsPerBlock);
                auto frameSpec = alpaka::onHost::FrameSpec{threadsPerBlock, blocks};
                queue.enqueue(exec, frameSpec, ElementwiseAddKernel<Exec>{}, a.getDeviceBuffer(), b.getDeviceBuffer(), result.getDeviceBuffer(), n);
                // Mark result device data modified and download
                result.markDeviceModified();
                result.toHost(queue);
                return result;
            }

            // High-level ReLU operation (simplified - CPU fallback for now)
            template<typename T, ::std::size_t Rank>
            Tensor<T, Rank> relu(const Tensor<T, Rank>& input) {
                
                // Create output tensor
                Tensor<T, Rank> result(input.shape(), input.dtype(), input.layout(), "relu_result");
                
                // Simple CPU implementation
                T* data_input = const_cast<T*>(input.hostData());
                T* data_result = result.hostData();
                
                ::std::size_t size = input.size();
                for(::std::size_t i = 0; i < size; ++i) {
                    data_result[i] = data_input[i] > T{0} ? data_input[i] : T{0};
                }
                
                return result;
            }

            // Device-dispatch ReLU
            template<typename T, ::std::size_t Rank, typename Exec, typename Device, typename Queue>
            Tensor<T, Rank> relu(Exec const& exec, Device const& device, Queue& queue, Tensor<T, Rank>& input) {
                input.ensureOnDevice(device, queue);
                Tensor<T, Rank> result(input.shape(), input.dtype(), input.layout(), "relu_result");
                result.allocateDevice(device);
                auto n = input.size();
                unsigned threadsPerBlock = 256u;
                unsigned blocks = static_cast<unsigned>((n + threadsPerBlock - 1) / threadsPerBlock);
                auto frameSpec = alpaka::onHost::FrameSpec{threadsPerBlock, blocks};
                queue.enqueue(exec, frameSpec, ElementwiseReluKernel<Exec>{}, input.getDeviceBuffer(), result.getDeviceBuffer(), n);
                result.markDeviceModified();
                result.toHost(queue);
                return result;
            }

            // In-place ReLU operation
            template<typename T, ::std::size_t Rank>
            void relu_inplace(Tensor<T, Rank>& tensor) {
                
                // Apply ReLU in-place on host data
                T* data = tensor.hostData();
                ::std::size_t size = tensor.size();
                
                for(::std::size_t i = 0; i < size; ++i) {
                    data[i] = data[i] > T{0} ? data[i] : T{0};
                }
            }

            // Device-dispatch in-place ReLU
            template<typename T, ::std::size_t Rank, typename Exec, typename Device, typename Queue>
            void relu_inplace(Exec const& exec, Device const& device, Queue& queue, Tensor<T, Rank>& tensor) {
                tensor.ensureOnDevice(device, queue);
                auto n = tensor.size();
                unsigned threadsPerBlock = 256u;
                unsigned blocks = static_cast<unsigned>((n + threadsPerBlock - 1) / threadsPerBlock);
                auto frameSpec = alpaka::onHost::FrameSpec{threadsPerBlock, blocks};
                // Reuse out=in by passing same buffer twice
                queue.enqueue(exec, frameSpec, ElementwiseReluKernel<Exec>{}, tensor.getDeviceBuffer(), tensor.getDeviceBuffer(), n);
                tensor.markDeviceModified();
                tensor.toHost(queue);
            }

        } // namespace ops
    } // namespace tensor
} // namespace alpaka
