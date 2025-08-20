/* Copyright 2025 Alpaka Tensor Library Contributors
 * SPDX-License-Identifier: MPL-2.0
 * 
 * For detailed documentation on Alpaka queue semantics and synchronization
 * patterns, see: include/alpaka/tensor/QueueSemantics.hpp
 */

#pragma once

#include <cstddef>

namespace alpaka::tensor {

// Forward declaration
template<typename T, std::size_t Rank> class Tensor;

template<typename T, std::size_t Rank>
class TensorView {
private:
    Tensor<T, Rank>* tensor_;

public:
    // Constructor from tensor reference
    explicit TensorView(Tensor<T, Rank>& tensor) 
        : tensor_(&tensor) {}

    // Get the underlying tensor
    Tensor<T, Rank>& getTensor() { return *tensor_; }
    const Tensor<T, Rank>& getTensor() const { return *tensor_; }

    // Auto-convert to tensor when needed (for backward compatibility)
    operator Tensor<T, Rank>&() { return *tensor_; }
    operator const Tensor<T, Rank>&() const { return *tensor_; }
    
    // Method chaining will be implemented via free functions in ops namespace
    // This keeps the view lightweight and avoids circular dependencies
};

} // namespace alpaka::tensor
