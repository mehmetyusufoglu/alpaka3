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
                template<typename TMdSpanA, typename TMdSpanB, typename TMdSpanResult>
                ALPAKA_FN_ACC void operator()(
                    TAcc const& acc,
                    TMdSpanA a,
                    TMdSpanB b,
                    TMdSpanResult result,
                    ::std::size_t numElements) const {
                    
                    for(auto [index] : alpaka::onAcc::makeIdxMap(
                        acc, 
                        alpaka::onAcc::worker::threadsInGrid, 
                        alpaka::IdxRange{numElements}))
                    {
                        result[index] = a[index] + b[index];
                    }
                }
            };

            // Elementwise ReLU kernel (activation function)
            template<typename TAcc>
            class ElementwiseReluKernel {
            public:
                template<typename TMdSpanInput, typename TMdSpanOutput>
                ALPAKA_FN_ACC void operator()(
                    TAcc const& acc,
                    TMdSpanInput input,
                    TMdSpanOutput output,
                    ::std::size_t numElements) const {
                    
                    for(auto [index] : alpaka::onAcc::makeIdxMap(
                        acc, 
                        alpaka::onAcc::worker::threadsInGrid, 
                        alpaka::IdxRange{numElements}))
                    {
                        auto value = input[index];
                        output[index] = value > static_cast<decltype(value)>(0) ? value : static_cast<decltype(value)>(0);
                    }
                }
            };

            // High-level addition operation (simplified - CPU fallback for now)
            template<typename T, ::std::size_t Rank>
            Tensor<T, Rank> add(const Tensor<T, Rank>& a, 
                               const Tensor<T, Rank>& b) {
                
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

        } // namespace ops
    } // namespace tensor
} // namespace alpaka
