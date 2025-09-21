# Alpaka Namespace Organization

This document provides a comprehensive overview of the namespace hierarchy in Alpaka and explains the purpose and usage of each namespace.

## Overview

Alpaka uses a hierarchical namespace structure to organize functionality across different abstraction layers and implementation details. The primary namespace is `alpaka`, with specialized sub-namespaces for different functional areas.

## Primary Namespace Structure

### `alpaka::`
The root namespace containing all Alpaka functionality.

## Host Operations - `alpaka::onHost::`

The `onHost` namespace contains all functionality for host-side operations - code that runs on the CPU to manage compute devices, memory, and execution.

### Core Host Components
- **Device Management**: Device selection, properties, and lifecycle
- **Memory Allocation**: Host-side memory management APIs
- **Queue Management**: Queue creation and synchronization
- **Execution Control**: Task submission and management

### Key Types in `onHost`
```cpp
// Device specification and selection
onHost::DeviceSpec<T_Api, T_DeviceKind>
onHost::DeviceSelector
onHost::Device

// Memory management
onHost::SharedBuffer
onHost::View

// Queue management  
onHost::Queue
onHost::Event
```

### `alpaka::onHost::cpu::`
Backend-specific implementation for CPU devices.
- Contains CPU-specific device, queue, and event implementations
- Used internally by the trait system for CPU backend

### `alpaka::onHost::trait::`
Host-side trait specializations for compile-time backend selection.
- Platform availability traits
- Device support traits
- API compatibility traits

### `alpaka::onHost::internal::`
Internal implementation details for host operations.
- Platform creation utilities
- Device property queries
- Memory allocation internals

### `alpaka::onHost::concepts::`
Concepts for host-side type checking.
- Device specification concepts
- Platform handle concepts

## Accelerator Operations - `alpaka::onAcc::`

The `onAcc` namespace contains functionality for accelerator-side operations - code that runs on compute devices (GPU kernels, etc.).

### Core Accelerator Components
- **Kernel Execution**: Thread indexing, synchronization
- **Shared Memory**: Dynamic and static shared memory
- **Iterators**: Multi-dimensional iteration patterns
- **Work Group Operations**: Block-level operations

### Key Types in `onAcc`
```cpp
// Thread and work group operations
onAcc::WorkGroup
onAcc::ThreadSpace
onAcc::Idx

// Memory iteration
onAcc::Iter
onAcc::IdxRange

// Shared memory
onAcc::SharedMem
```

### `alpaka::onAcc::trait::`
Accelerator-side trait specializations.
- Iterator traversal traits
- Shared memory traits
- Synchronization traits

### `alpaka::onAcc::internal::`
Internal accelerator implementation details.
- Iterator implementation
- Work group internals
- Thread space management

### `alpaka::onAcc::concepts::`
Concepts for accelerator-side type checking.
- Iterator concepts
- Work group concepts

## API Abstraction - `alpaka::api::`

The `api` namespace organizes backend-specific programming model implementations.

### Backend API Namespaces

#### `alpaka::api::host::`
CPU/OpenMP backend implementation.
- Host threading models
- CPU-specific optimizations

#### `alpaka::api::cuda::`
NVIDIA CUDA backend implementation.
- CUDA-specific device management
- CUDA memory models

#### `alpaka::api::hip::`
AMD HIP backend implementation.
- HIP-specific device management
- AMD GPU optimizations

#### `alpaka::api::oneApi::`
Intel oneAPI/SYCL backend implementation.
- SYCL device selectors
- Intel GPU optimizations

### API Trait System
Each API namespace contains trait specializations for:
- Platform availability
- Device compatibility
- Memory allocation strategies
- Kernel launch mechanisms

## Device Kind Classification - `alpaka::deviceKind::`

Device kinds represent hardware categories independent of programming APIs.

### Device Kind Types
```cpp
deviceKind::Cpu        // CPU devices
deviceKind::NvidiaGpu  // NVIDIA GPUs  
deviceKind::AmdGpu     // AMD GPUs
deviceKind::IntelGpu   // Intel GPUs
```

### Device Kind Organization
- **Base Classes**: Common device kind interfaces
- **Traits**: Device capability traits
- **Concepts**: Type checking for device kinds

## Executor Specification - `alpaka::exec::`

Executors define how parallelism is organized on devices.

### Executor Types
```cpp
exec::cpuSerial   // Serial CPU execution
exec::ompBlocks   // OpenMP block-level parallelism
exec::gpuCuda     // CUDA GPU execution
exec::gpuHip      // HIP GPU execution  
exec::oneApi      // oneAPI/SYCL execution
```

## Memory Management Namespaces

### `alpaka::mem::`
Core memory abstractions and utilities.
- **MdSpan**: Multi-dimensional span interface
- **View**: Memory view abstraction
- **Alignment**: Memory alignment specifications
- **Iterators**: Memory traversal patterns

### Memory Trait Specializations
- Allocation strategies per backend
- Memory view traits
- Alignment requirements

## Core Infrastructure Namespaces

### `alpaka::trait::`
Fundamental trait system for compile-time polymorphism.
- **Type Traits**: Basic type information
- **Dimension Traits**: Multi-dimensional support
- **Kernel Argument Traits**: Parameter passing

### `alpaka::concepts::`
C++20 concepts for type checking.
- **API Concepts**: Backend API validation
- **Device Concepts**: Device type validation
- **Memory Concepts**: Memory type validation
- **Vector Concepts**: Multi-dimensional vector validation

### `alpaka::internal::`
Internal implementation details (not for user consumption).
- **Interface Utilities**: Internal API helpers
- **Type Manipulation**: Template metaprogramming utilities
- **Platform Internals**: Low-level platform operations

### `alpaka::detail::`
Implementation details for specific components.
- **Base Classes**: Common base class implementations
- **Helper Utilities**: Internal helper functions

## Specialized Namespaces

### Backend-Specific Unification
- **`alpaka::unifiedCudaHip::`**: Shared CUDA/HIP implementations
- **`alpaka::syclGeneric::`**: Generic SYCL implementations

### Testing and Examples
- **`alpaka::example::`**: Example code utilities
- **`alpaka::test::`**: Testing framework integration

## Namespace Usage Patterns

### 1. User Code Structure
```cpp
#include <alpaka/alpaka.hpp>

using namespace alpaka;

// Host-side: Device and memory management
auto deviceSpec = onHost::DeviceSpec{api::cuda, deviceKind::nvidiaGpu};
auto device = onHost::makeDeviceSelector(deviceSpec).makeDevice(0);
auto buffer = onHost::alloc<float>(device, Vec{1024});

// Accelerator-side: Kernel implementation
struct MyKernel {
    template<typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc) const {
        auto idx = onAcc::getIdx(acc);
        // Kernel logic...
    }
};
```

### 2. Backend Specialization
```cpp
// API-specific implementations
namespace alpaka::api::cuda {
    // CUDA-specific optimizations
}

// Device-kind specific optimizations  
if constexpr(TDeviceKind{} == deviceKind::nvidiaGpu) {
    // NVIDIA-specific logic
}
```

### 3. Trait Specializations
```cpp
// Host-side trait specialization
namespace alpaka::onHost::trait {
    template<>
    struct IsDeviceSupportedBy::Op<deviceKind::NvidiaGpu, api::Cuda> 
        : std::true_type {};
}

// Accelerator-side trait specialization
namespace alpaka::onAcc::trait {
    template<typename TAcc>
    struct GetSharedMemDynSizeBytes::Op<MyKernel, TAcc> {
        // Shared memory size calculation
    };
}
```

## Summary

The Alpaka namespace organization follows a clear hierarchical structure:

1. **`alpaka::`** - Root namespace
2. **`alpaka::onHost::`** - Host-side operations (device management, memory allocation)
   - `cpu`
   - `trait`
   - `internal`
   - `concepts`
3. **`alpaka::onAcc::`** - Accelerator-side operations (kernel execution, shared memory)
   - `trait`
   - `internal`
   - `concepts`
   - `idxTrait`
   - `range`
   - `detail`
4. **`alpaka::api::`** - Backend-specific programming model implementations
   - `host`
   - `cuda`
   - `hip`
   - `oneApi`
   - `util`
5. **`alpaka::deviceKind::`** - Hardware category classification
   - `detail`
   - `trait`
   - `concepts`
6. **`alpaka::exec::`** - Execution strategy specification
   - `trait`
   - `internal`
7. **`alpaka::trait::`** - Compile-time polymorphism system
8. **`alpaka::concepts::`** - Type validation and constraints
9. **`alpaka::internal::`** - Implementation details (internal use only)
   - `concepts`

This organization enables:
- **Clear separation** between host and accelerator code
- **Compile-time backend selection** through traits
- **Type safety** through concepts
- **Extensibility** for new backends and device types
- **Performance optimization** through specialization