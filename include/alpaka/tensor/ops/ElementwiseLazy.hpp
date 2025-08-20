/* Copyright 2025 Alpaka Tensor Library Contributors
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <alpaka/tensor/TensorCore.hpp>
#include <alpaka/tensor/TensorView.hpp>
#include <alpaka/tensor/ops/ElementwiseGeneric.hpp>

namespace alpaka::tensor::ops {

// View-based operations that don't auto-sync to host
// These operations stay on device until explicitly synced

// out = f(a, b) - lazy version
template<typename T, std::size_t Rank, typename Exec, typename Device, typename Queue, typename Functor>
Tensor<T, Rank> binaryLazy(Exec const& exec, Device& device, Queue& queue, Tensor<T, Rank>& a, Tensor<T, Rank>& b, Functor f, const char* name="binary") {
    a.ensureOnDevice(device, queue);
    b.ensureOnDevice(device, queue);
    Tensor<T, Rank> out(a.shape(), a.dtype(), a.layout(), name);
    out.allocateDevice(device);
    auto n = a.size();
    auto frame = detail::makeFrame<Exec, Queue>(n);
    queue.enqueue(exec, frame, BinaryKernel{}, a.getDeviceBuffer(device), b.getDeviceBuffer(device), out.getDeviceBuffer(device), n, f);
    ::alpaka::onHost::wait(queue);  // Wait for kernel completion
    out.markDeviceModified();
    // Note: NO toHost() call - stays on device!
    return out;
}

// out = f(in) - lazy version
template<typename T, std::size_t Rank, typename Exec, typename Device, typename Queue, typename Functor>
Tensor<T, Rank> unaryLazy(Exec const& exec, Device& device, Queue& queue, Tensor<T, Rank>& in, Functor f, const char* name="unary") {
    in.ensureOnDevice(device, queue);
    Tensor<T, Rank> out(in.shape(), in.dtype(), in.layout(), name);
    out.allocateDevice(device);
    auto n = in.size();
    auto frame = detail::makeFrame<Exec, Queue>(n);
    queue.enqueue(exec, frame, UnaryKernel{}, in.getDeviceBuffer(device), out.getDeviceBuffer(device), n, f);
    ::alpaka::onHost::wait(queue);  // Wait for kernel completion
    out.markDeviceModified();
    // Note: NO toHost() call - stays on device!
    return out;
}

// Lazy versions of common operations
template<typename T, std::size_t Rank, typename Exec, typename Device, typename Queue>
Tensor<T, Rank> addLazy(Exec const& exec, Device& device, Queue& queue, Tensor<T, Rank>& a, Tensor<T, Rank>& b) {
    return binaryLazy<T, Rank>(exec, device, queue, a, b, AddOp{}, "add_lazy");
}

template<typename T, std::size_t Rank, typename Exec, typename Device, typename Queue>
Tensor<T, Rank> mulLazy(Exec const& exec, Device& device, Queue& queue, Tensor<T, Rank>& a, Tensor<T, Rank>& b) {
    return binaryLazy<T, Rank>(exec, device, queue, a, b, MulOp{}, "mul_lazy");
}

template<typename T, std::size_t Rank, typename Exec, typename Device, typename Queue, typename S>
Tensor<T, Rank> mul_scalarLazy(Exec const& exec, Device& device, Queue& queue, Tensor<T, Rank>& in, S scalar) {
    return unaryLazy<T, Rank>(exec, device, queue, in, MulScalarOp<S>{scalar}, "mul_scalar_lazy");
}

template<typename T, std::size_t Rank, typename Exec, typename Device, typename Queue>
Tensor<T, Rank> reluLazy(Exec const& exec, Device& device, Queue& queue, Tensor<T, Rank>& in) {
    return unaryLazy<T, Rank>(exec, device, queue, in, ReluOp{}, "relu_lazy");
}

// =============================================================================
// VIEW-BASED OPERATIONS - These work with TensorView for method chaining
// =============================================================================

// Operations on TensorView that return TensorView for chaining
template<typename T, std::size_t Rank, typename Exec, typename Device, typename Queue, typename S>
TensorView<T, Rank> mul_scalar(TensorView<T, Rank> view, Exec const& exec, Device& device, Queue& queue, S scalar) {
    auto& tensor = view.getTensor();
    auto result = mul_scalarLazy<T, Rank>(exec, device, queue, tensor, scalar);
    tensor = std::move(result);  // Update the view's tensor
    return view;  // Return the same view for chaining
}

template<typename T, std::size_t Rank, typename Exec, typename Device, typename Queue>
TensorView<T, Rank> add(TensorView<T, Rank> view, Exec const& exec, Device& device, Queue& queue, Tensor<T, Rank>& other) {
    auto& tensor = view.getTensor();
    auto result = addLazy<T, Rank>(exec, device, queue, tensor, other);
    tensor = std::move(result);
    return view;
}

// Final operation that syncs and returns tensor
template<typename T, std::size_t Rank, typename Exec, typename Device, typename Queue>
Tensor<T, Rank> relu(TensorView<T, Rank> view, Exec const& exec, Device& device, Queue& queue) {
    auto& tensor = view.getTensor();
    auto result = reluLazy<T, Rank>(exec, device, queue, tensor);
    result.toHost(device, queue);  // Auto-sync at end of chain
    return result;
}

// =============================================================================
// SYNTAX SUGAR - Allow method-like chaining on tensors
// =============================================================================

// Free functions that enable: tensor.view().mul_scalar(...).relu(...)
template<typename T, std::size_t Rank, typename Exec, typename Device, typename Queue, typename S>
auto operator|(TensorView<T, Rank> view, std::tuple<Exec, Device&, Queue&, S>) -> TensorView<T, Rank> {
    // This is a placeholder - we'll implement proper chaining syntax later
    return view;
}

} // namespace alpaka::tensor::ops
