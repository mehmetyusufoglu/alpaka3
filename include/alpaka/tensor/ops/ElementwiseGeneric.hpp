#pragma once
#include <alpaka/alpaka.hpp>
#include <alpaka/tensor/TensorGPUFixed.hpp>
#include <type_traits>

namespace alpaka { namespace tensor { namespace ops {

// Generic unary elementwise kernel applying Functor: out[i] = f(in[i])
class UnaryKernel {
public:
    template<typename Acc, typename InBuf, typename OutBuf, typename Functor>
    ALPAKA_FN_ACC void operator()(Acc const& acc, InBuf in, OutBuf out, std::size_t n, Functor f) const {
        for(auto [i] : alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{n})) {
            out[i] = f(in[i]);
        }
    }
};

// Generic binary elementwise kernel: out[i] = f(a[i], b[i])
class BinaryKernel {
public:
    template<typename Acc, typename ABuf, typename BBuf, typename OutBuf, typename Functor>
    ALPAKA_FN_ACC void operator()(Acc const& acc, ABuf a, BBuf b, OutBuf out, std::size_t n, Functor f) const {
        for(auto [i] : alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{n})) {
            out[i] = f(a[i], b[i]);
        }
    }
};

// Helper launching functions
namespace detail {
    template<typename Exec, typename Queue>
    inline auto makeFrame(std::size_t n) {
        unsigned threadsPerBlock = 256u;
        unsigned blocks = static_cast<unsigned>((n + threadsPerBlock - 1) / threadsPerBlock);
        return alpaka::onHost::FrameSpec{alpaka::Vec<unsigned int,1u>{blocks}, alpaka::Vec<unsigned int,1u>{threadsPerBlock}};
    }
}

// Public APIs ---------------------------------------------------------

// out = a (+,-,*,/) b using provided functor f
// Functor must be __host__ __device__ compatible (ALPAKA_FN_HOST_ACC)

template<typename T, std::size_t Rank, typename Exec, typename Device, typename Queue, typename Functor>
Tensor<T, Rank> binary(Exec const& exec, Device& device, Queue& queue, Tensor<T, Rank>& a, Tensor<T, Rank>& b, Functor f, const char* name="binary") {
    a.ensureOnDevice(device, queue);
    b.ensureOnDevice(device, queue);
    Tensor<T, Rank> out(a.shape(), a.dtype(), a.layout(), name);
    out.allocateDevice(device);
    auto n = a.size();
    auto frame = detail::makeFrame<Exec, Queue>(n);
    queue.enqueue(exec, frame, BinaryKernel{}, a.getDeviceBuffer(device), b.getDeviceBuffer(device), out.getDeviceBuffer(device), n, f);
    out.markDeviceModified();
    out.toHost(device, queue);
    return out;
}

// out = f(in)
template<typename T, std::size_t Rank, typename Exec, typename Device, typename Queue, typename Functor>
Tensor<T, Rank> unary(Exec const& exec, Device& device, Queue& queue, Tensor<T, Rank>& in, Functor f, const char* name="unary") {
    in.ensureOnDevice(device, queue);
    Tensor<T, Rank> out(in.shape(), in.dtype(), in.layout(), name);
    out.allocateDevice(device);
    auto n = in.size();
    auto frame = detail::makeFrame<Exec, Queue>(n);
    queue.enqueue(exec, frame, UnaryKernel{}, in.getDeviceBuffer(device), out.getDeviceBuffer(device), n, f);
    out.markDeviceModified();
    out.toHost(device, queue);
    return out;
}

// Convenience wrappers for common operations

struct AddOp { template<typename V> ALPAKA_FN_HOST_ACC V operator()(V a, V b) const { return a + b; } };
struct SubOp { template<typename V> ALPAKA_FN_HOST_ACC V operator()(V a, V b) const { return a - b; } };
struct MulOp { template<typename V> ALPAKA_FN_HOST_ACC V operator()(V a, V b) const { return a * b; } };
struct DivOp { template<typename V> ALPAKA_FN_HOST_ACC V operator()(V a, V b) const { return a / b; } };
struct ReluOp { template<typename V> ALPAKA_FN_HOST_ACC V operator()(V v) const { return v > V{} ? v : V{}; } };
// Scalar ops wrappers (functors capturing scalar value)
template<typename S>
struct AddScalarOp { S s; template<typename V> ALPAKA_FN_HOST_ACC V operator()(V a) const { return a + static_cast<V>(s); } };
template<typename S>
struct MulScalarOp { S s; template<typename V> ALPAKA_FN_HOST_ACC V operator()(V a) const { return a * static_cast<V>(s); } };

// High-level wrappers

template<typename T, std::size_t Rank, typename Exec, typename Device, typename Queue>
Tensor<T, Rank> add(Exec const& exec, Device& device, Queue& queue, Tensor<T, Rank>& a, Tensor<T, Rank>& b) {
    return binary<T, Rank>(exec, device, queue, a, b, AddOp{}, "add");
}

template<typename T, std::size_t Rank, typename Exec, typename Device, typename Queue>
Tensor<T, Rank> sub(Exec const& exec, Device& device, Queue& queue, Tensor<T, Rank>& a, Tensor<T, Rank>& b) {
    return binary<T, Rank>(exec, device, queue, a, b, SubOp{}, "sub");
}

template<typename T, std::size_t Rank, typename Exec, typename Device, typename Queue>
Tensor<T, Rank> mul(Exec const& exec, Device& device, Queue& queue, Tensor<T, Rank>& a, Tensor<T, Rank>& b) {
    return binary<T, Rank>(exec, device, queue, a, b, MulOp{}, "mul");
}

template<typename T, std::size_t Rank, typename Exec, typename Device, typename Queue>
Tensor<T, Rank> div(Exec const& exec, Device& device, Queue& queue, Tensor<T, Rank>& a, Tensor<T, Rank>& b) {
    return binary<T, Rank>(exec, device, queue, a, b, DivOp{}, "div");
}

template<typename T, std::size_t Rank, typename Exec, typename Device, typename Queue>
Tensor<T, Rank> relu(Exec const& exec, Device& device, Queue& queue, Tensor<T, Rank>& in) {
    return unary<T, Rank>(exec, device, queue, in, ReluOp{}, "relu");
}

// sub/mul/div already provided via generic wrappers above

template<typename T, std::size_t Rank, typename Exec, typename Device, typename Queue, typename S>
Tensor<T, Rank> add_scalar(Exec const& exec, Device& device, Queue& queue, Tensor<T, Rank>& in, S scalar, const char* name="add_scalar") {
    return unary<T, Rank>(exec, device, queue, in, AddScalarOp<S>{scalar}, name);
}

template<typename T, std::size_t Rank, typename Exec, typename Device, typename Queue, typename S>
Tensor<T, Rank> mul_scalar(Exec const& exec, Device& device, Queue& queue, Tensor<T, Rank>& in, S scalar, const char* name="mul_scalar") {
    return unary<T, Rank>(exec, device, queue, in, MulScalarOp<S>{scalar}, name);
}

// In-place ReLU

template<typename T, std::size_t Rank, typename Exec, typename Device, typename Queue>
void relu_inplace(Exec const& exec, Device& device, Queue& queue, Tensor<T, Rank>& t) {
    t.ensureOnDevice(device, queue);
    auto n = t.size();
    auto frame = detail::makeFrame<Exec, Queue>(n);
    queue.enqueue(exec, frame, UnaryKernel{}, t.getDeviceBuffer(device), t.getDeviceBuffer(device), n, ReluOp{});
    t.markDeviceModified();
    t.toHost(device, queue);
}

}}} // namespaces
