# Alpaka Tensor Library — Architecture Overview

This document summarizes the design of the Alpaka Tensor Library, its core abstractions, execution model, and provider ecosystem.

- Strongly-typed, rank-static tensors: `Tensor<T, Rank, Device>` with 1D–4D aliases
- Explicit coherence state tracking between host and device copies
- Views as lightweight wrappers: `TensorView<T, Rank, Device>`
- High-level ops layered on Alpaka execution (queues/executors)
- Provider model for vendor-accelerated backends (cuBLAS, cuDNN, rocBLAS, MIOpen)

## Core Types

- Tensor: owns host/device buffers; shape and extents stored; lazy device allocation
- CoherenceState: Unallocated, HostFresh, DeviceFresh, BothFresh
- Helpers: factories (`makeTensor{1D..4D}`), host fillers, context bundling

## Tensor internals (storage model)

- Host storage: an owning buffer allocated via `alpaka::onHost::allocHost<T>(extents)`
  - Internally, `Tensor` holds `std::optional<HostBuffer>` and materializes it on first use
  - Access via `hostData()` returns a raw pointer; call `markHostModified()` after writes
- Device storage: an owning buffer allocated lazily via `alpaka::onHost::alloc<T>(device, extents)`
  - Held as `std::optional<DeviceBuffer>` and bound to the constructing device
  - Access via `deviceBuffer(device, queue)` (ensures allocation and synchronizes if host was fresher)
- No `std::mdspan` internally: the class stores raw contiguous buffers (host/device) plus shape/extent metadata
  - Multi-dimensional access for kernels is achieved through Alpaka MdSpan concepts on the buffer views, or via index maps
- Coherence state machine:
  - `HostFresh` → `ensureOnDevice()` copies host→device and moves to `BothFresh`
  - `DeviceFresh` → `toHost()` copies device→host and moves to `BothFresh`
  - `markHostModified()`/`markDeviceModified()` set the fresh side after mutations

## Execution Model

- All ops accept `(exec, device, queue, ...)`
- Enqueue work on queue; most ops avoid immediate wait; host access triggers `toHost()` or explicit wait
- FrameSpec used to shape launch geometry; kernels often use `onAcc::makeIdxMap`

## Providers

- `IOpProvider` abstract base with OpType and `*_status` virtuals
- `ProviderRegistry` selects providers per exec at compile time
- DefaultProvider delegates to generic Alpaka kernels
- CleanTensorOpContext coordinates providers and offers typed convenience methods

## Directory structure (include/alpaka/tensor)

- `tensor.hpp` — umbrella include that aggregates all tensor components
- `TensorCore.hpp` — main `Tensor<T, Rank, Device>` implementation (storage, coherence, allocation)
- `TensorView.hpp` — lightweight non-owning wrapper around `Tensor`
- `TensorGeneric.hpp` — rank-agnostic helpers and example generic kernels (e.g., bias add)
- `TensorDescriptor.hpp` — debug-time layout/dtype assertions and descriptors
- `Helpers.hpp` — factories (`makeTensor*`), host fill utilities, convenience aliases
- `QueueSemantics.hpp` — documentation for queue ordering and sync
- `ops/` — operation implementations
  - `Conv2D.hpp` — forward conv; launches naive or tiled kernels; provider fallback
  - `Gemm.hpp` — GEMM; cuBLAS fast path, generic kernel fallback
  - `Activations.hpp` — ReLU (SIMD and generic), in-place support
  - `InferenceOps.hpp` — BiasAdd, Linear(+bias), Softmax, Flatten, pooling shells
  - `Layer.hpp` — composable high-level layers (Conv2D, BN, ReLU, Pool, Residual blocks)
  - `ops/kernels/` — device kernels for ops (Conv2D, Pooling, GEMM, BatchNorm, Softmax)
  - `ops/layers/` — grouped layer definitions
- `providers/` — vendor libraries integration
  - `ProviderInterface.hpp` — `IOpProvider` + concepts and status shims
  - `ProviderRegistry.hpp` — compile-time selection and factory methods
  - `{CuBLAS, CuDNN, RocBLAS, MIOpen}Provider.hpp` — typed backends when available
  - `DefaultProvider.hpp` — generic Alpaka fallback implementations
- `CleanTensorOpContext.hpp` — orchestrates provider selection and fallbacks
- `SyncDebug.hpp` — debugging aids for synchronization

Kernels directory importance:
- `ops/kernels/` contains the device-side functors used by high-level ops
- Encapsulates backend-agnostic kernels (iterate with `onAcc::makeIdxMap`) and specialized tiled variants
- Allows ops to choose between naive and optimized kernels based on problem size/backend capabilities

## High-level operations and alternatives

- High-level wrappers (`alpaka::tensor::highlevel`) provide ergonomic entry points:
  - `highlevel::gemm`, `highlevel::conv2d`, `highlevel::relu`, `highlevel::relu_inplace`
  - They forward to `ops::*` (generic implementations) or provider-backed paths via context
- If you don’t use HighLevel APIs:
  - Use low-level ops directly in `alpaka::tensor::ops` (e.g., `ops::conv2d`, `ops::gemm`, `ops::relu`)
  - Or, for vendor libraries when available, call through `CleanTensorOpContext` for provider selection and fallbacks
- When to choose which:
  - HighLevel: fastest to wire, consistent defaults; suitable for examples and common cases
  - Low-level ops: more control over launch, parameters, and chaining; minimal overhead
  - Provider context: when you want the best available backend (cuBLAS/cuDNN/MIOpen) with automatic fallback

See detailed docs in:
- `api-surface.md`
- `memory-and-layout.md`
- `execution-and-providers.md`
- `ops-catalog.md`
- `practical-usage.md`
