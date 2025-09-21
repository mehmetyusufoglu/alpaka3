# Alpaka Memory System Architecture

## Overview

This document provides a comprehensive analysis of Alpaka's memory management system, which abstracts memory allocation, access patterns, and data views across different compute backends (CPU, CUDA, HIP, OneAPI/SYCL). The memory system is designed around the principle of **backend-agnostic memory abstractions** that allow unified memory operations while providing backend-specific optimizations.

## Core Design Principles

### 1. Unified Memory Interface
- **Single API** for all memory operations across backends
- **Type-safe** memory allocations and transfers
- **RAII-based** memory management with automatic cleanup
- **Alignment-aware** allocations for optimal performance

### 2. Memory Type Abstraction
Alpaka provides multiple memory types to match hardware capabilities:
- **Device Memory**: Backend-specific optimal memory (GPU global memory, CPU heap)
- **Host Memory**: CPU-accessible memory
- **Unified/Managed Memory**: Accessible from both host and device
- **Mapped/Pinned Memory**: Host memory mapped to device address space
- **Shared Memory**: Block-level fast memory within kernels

### 3. View-Based Access
- **Non-owning views** separate data access from memory ownership
- **Multi-dimensional indexing** with automatic pitch/stride handling
- **Const-correctness** propagation through view hierarchies

## Memory Allocation System

### 1. Core Allocation Functions

Alpaka provides a comprehensive set of allocation functions in `onHost::` namespace:

```cpp
// Device memory allocation
template<typename T_Type>
auto alloc(concepts::Device auto const& device, 
           alpaka::concepts::VectorOrScalar auto const& extents);

// Host memory allocation  
template<typename T_ValueType>
auto allocHost(alpaka::concepts::VectorOrScalar auto const& extents);

// Unified memory (accessible from host and device)
template<typename T_Type>
auto allocUnified(concepts::Device auto const& device,
                  alpaka::concepts::VectorOrScalar auto const& extents);

// Pinned/mapped memory (host memory accessible by device)
template<typename T_Type>
auto allocMapped(concepts::Device auto const& device,
                 alpaka::concepts::VectorOrScalar auto const& extents);

// Deferred allocation (memory available after queue operations)
template<typename T_Type>
auto allocDeferred(Queue auto const& queue,
                   alpaka::concepts::VectorOrScalar auto const& extents);
```

### 2. Backend-Specific Implementation

Each backend implements allocation through the trait system:

| **Backend** | **Device Memory** | **Unified Memory** | **Mapped Memory** | **Implementation** |
|-------------|-------------------|--------------------|--------------------|-------------------|
| **Host CPU** | `alignedAlloc()` | Same as device | Same as device | `api/host/Device.hpp` |
| **CUDA** | `cudaMalloc()`/`cudaMallocPitch()` | `cudaMallocManaged()` | `cudaHostMalloc()` | `api/unifiedCudaHip/Device.hpp` |
| **HIP** | `hipMalloc()`/`hipMallocPitch()` | `hipMallocManaged()` | `hipHostMalloc()` | `api/unifiedCudaHip/Device.hpp` |
| **OneAPI** | `sycl::aligned_alloc_device()` | `sycl::aligned_alloc_shared()` | `sycl::aligned_alloc_host()` | `api/syclGeneric/Device.hpp` |

### 3. Memory Alignment and Optimization

```cpp
// Backend-specific alignment optimization
constexpr uint32_t alignment = api::util::simdOptimizedAlignment<T_Type>(api, deviceKind);

// Pitch calculation for optimal memory access
auto [memSizeInByte, pitches] = api::util::emulatedAlignedMemDescription<T_Type>(alignment, extents);
```

**Alignment Specifications:**
- **Host CPU**: SIMD-optimized alignment (typically 32-64 bytes)
- **CUDA/HIP**: 128-256 byte alignment for coalesced access
- **OneAPI**: Variable alignment based on device capabilities

### 4. RAII Memory Management

All allocations return `SharedBuffer` objects with automatic cleanup:

```cpp
auto sharedBuffer = onHost::SharedBuffer{
    deviceDependency,          // Device lifetime management
    ptr,                       // Raw pointer to memory
    extents,                   // Dimensions
    pitches,                   // Memory layout information
    std::move(deleter),        // Custom deleter for backend
    Alignment<alignment>{}     // Alignment information
};
```

## View and MdSpan System

### 1. MdSpan: Multi-Dimensional Data Access

`MdSpan` provides a lightweight, non-owning view to multi-dimensional data:

```cpp
template<
    typename T_Type,                    // Element type (const = read-only)
    concepts::Vector T_Extents,         // Dimensions
    concepts::Vector T_Pitches,         // Memory layout
    concepts::Alignment T_MemAlignment  // Memory alignment
>
struct MdSpan {
    using value_type = T_Type;
    using reference = value_type&;
    using pointer = value_type*;
    
    // Multi-dimensional indexing
    constexpr reference operator[](concepts::Vector auto const& idx);
    
    // Iterator support
    constexpr auto begin() const;
    constexpr auto end() const;
    
    // Metadata access
    constexpr auto getExtents() const;
    constexpr auto getPitches() const;
    constexpr auto data() const;
};
```

### 2. View: API-Aware Memory Views

`View` extends `MdSpan` with API information for backend-specific operations:

```cpp
template<
    typename T_Api,                     // Backend API (host, cuda, hip, oneApi)
    typename T_Type,                    // Element type
    alpaka::concepts::Vector T_Extents, // Dimensions
    alpaka::concepts::Alignment T_MemAlignment = Alignment<>
>
struct View : MdSpan<T_Type, typename T_Extents::UniVec, typename T_Extents::UniVec, T_MemAlignment> {
    // API identification
    static consteval T_Api getApi();
    
    // Sub-view creation
    constexpr auto getSubView(auto const& extents);
    
    // Const view creation
    constexpr auto getConstView() const;
    
    // MdSpan conversion
    constexpr alpaka::concepts::MdSpan auto getMdSpan();
};
```

### 3. View Creation and Usage

```cpp
// From allocated buffer
auto buffer = onHost::alloc<float>(device, Vec{1024, 768});
auto view = makeView(buffer);  // Automatic API and layout detection

// From raw pointer
float* rawPtr = getRawPointer();
auto view = makeView(api::cuda, rawPtr, Vec{width, height});

// From standard containers
std::vector<int> hostData(1000);
auto view = makeView(hostData);  // Automatically api::host
```

### 4. Pitch and Stride Handling

Alpaka automatically handles memory layout complexities:

```cpp
// Automatic pitch calculation for optimal memory access
auto pitches = alpaka::calculatePitchesFromExtents<T_Type>(extents);

// Manual pitch specification for custom layouts
auto view = makeView(api, pointer, extents, customPitches);
```

## Shared Memory System

### 1. Static Shared Memory

Block-level static shared memory allocation within kernels:

```cpp
// In kernel code with ALPAKA_FN_ACC
template<typename T, size_t T_uniqueId>
constexpr decltype(auto) declareSharedMdArray(
    concepts::Acc auto const& acc,
    alpaka::concepts::CVector auto const& extent)
{
    return acc[layer::shared].template allocVar<T, T_uniqueId>();
}

// Usage in kernel
ALPAKA_FN_ACC void operator()(auto const& acc, ...) const {
    // 2D shared memory array: 4 columns × 3 rows
    auto sharedArray = declareSharedMdArray<float, uniqueId()>(acc, CVec<uint32_t, 3, 4>{});
    
    // Single shared variable
    auto& sharedVar = declareSharedVar<int, uniqueId()>(acc);
}
```

### 2. Dynamic Shared Memory

Runtime-configurable shared memory allocation:

```cpp
// Method 1: Kernel member variable
struct MyKernel {
    uint32_t dynSharedMemBytes = 1024;  // Size in bytes
    
    ALPAKA_FN_ACC void operator()(auto const& acc, ...) const {
        auto* dynMem = getDynSharedMem<float>(acc);
        // Use dynMem as array of floats
    }
};

// Method 2: Trait specialization
namespace alpaka::onHost::trait {
    template<typename T_FrameSpec>
    struct BlockDynSharedMemBytes<MyKernel, T_FrameSpec> {
        uint32_t operator()(auto const executor, auto const&... args) const {
            return 2048;  // Dynamic calculation based on parameters
        }
    };
}
```

### 3. Backend-Specific Shared Memory Implementation

| **Backend** | **Static Shared Memory** | **Dynamic Shared Memory** | **Implementation** |
|-------------|---------------------------|----------------------------|--------------------|
| **Host CPU** | Thread-local storage via `SharedStorage` | Same storage pool | `api/host/block/mem/` |
| **OpenMP** | OpenMP thread-local shared storage | Barrier-synchronized allocation | `api/host/block/mem/OmpStaticShared.hpp` |
| **CUDA** | `__shared__` memory | `extern __shared__` memory | Native CUDA shared memory |
| **HIP** | `__shared__` memory | `extern __shared__` memory | Native HIP shared memory |
| **OneAPI** | `sycl::local_accessor` | Dynamic `sycl::local_accessor` | `api/oneApi/Queue.hpp` |

### 4. Shared Memory Access Patterns

```cpp
ALPAKA_FN_ACC void operator()(auto const& acc, ...) const {
    // Block-wide shared memory
    auto sharedBuffer = declareSharedMdArray<float, 0>(acc, CVec<uint32_t, 256>{});
    
    // Thread-safe initialization
    auto idx = getIdxWithin<Block>(acc);
    if (idx[0] == 0) {
        // Initialize shared memory from thread 0
        sharedBuffer[0] = initialValue;
    }
    onAcc::syncBlockThreads(acc);  // Synchronize before use
    
    // All threads can now safely access initialized shared memory
    auto value = sharedBuffer[threadIdx];
}
```

## Kernel Memory Usage Patterns

### 1. View/MdSpan in Kernel Parameters

Views and MdSpans are **fully supported** in kernel code:

```cpp
class VectorAddKernel {
    ALPAKA_FN_ACC auto operator()(
        auto const& acc,
        alpaka::concepts::MdSpan auto const A,    // Input view
        alpaka::concepts::MdSpan auto const B,    // Input view  
        alpaka::concepts::MdSpan auto C,          // Output view
        auto const& numElements) const -> void
    {
        // Direct multi-dimensional indexing
        for (auto idx : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, 
                                          IdxRange{numElements})) {
            C[idx] = A[idx] + B[idx];
        }
    }
};
```

### 2. Kernel Memory Access Patterns

```cpp
// 1D linear access pattern
ALPAKA_FN_ACC void linearKernel(auto const& acc, concepts::MdSpan auto data) {
    auto globalIdx = getIdxWithin<Grid>(acc);
    if (globalIdx < data.getExtents()) {
        data[globalIdx] = computeValue(globalIdx);
    }
}

// 2D/3D structured access pattern  
ALPAKA_FN_ACC void structuredKernel(auto const& acc, concepts::MdSpan auto data) {
    auto blockIdx = getIdxWithin<Grid>(acc);
    auto threadIdx = getIdxWithin<Block>(acc);
    
    auto globalIdx = blockIdx * getExtentsOf<Block>(acc) + threadIdx;
    if ((globalIdx < data.getExtents()).reduce(std::logical_and{})) {
        data[globalIdx] = processElement(globalIdx);
    }
}

// SIMD-optimized access pattern
ALPAKA_FN_ACC void simdKernel(auto const& acc, concepts::MdSpan auto out,
                              concepts::MdSpan auto in0, concepts::MdSpan auto in1) {
    auto simdGrid = onAcc::SimdAlgo{onAcc::worker::threadsInGrid};
    simdGrid.concurrent(acc, out.getExtents(),
        [](auto const&, auto&& o, auto&& i0, auto&& i1) constexpr {
            o = i0.load() + i1.load();  // Vectorized operation
        }, out, in0, in1);
}
```

### 3. Memory Type Compatibility

| **Memory Type** | **Host Access** | **Device Access** | **Kernel Parameter** | **Performance** |
|-----------------|-----------------|-------------------|----------------------|-----------------|
| **Device Memory** | ❌ | ✅ | ✅ | Optimal |
| **Host Memory** | ✅ | ❌ | ✅ (via transfer) | Requires copy |
| **Unified Memory** | ✅ | ✅ | ✅ | Good (with migration) |
| **Mapped Memory** | ✅ | ✅ | ✅ | Moderate (PCIe overhead) |
| **Shared Memory** | ❌ | ✅ (block-local) | ❌ | Fastest (on-chip) |

### 4. View Const-Correctness in Kernels

```cpp
ALPAKA_FN_ACC void kernel(auto const& acc,
                          concepts::MdSpan auto const readonly,   // const MdSpan = read-only data
                          concepts::MdSpan auto readwrite) const  // non-const MdSpan = mutable data
{
    // Compile-time enforced read-only access
    auto value = readonly[idx];  // ✅ Valid
    // readonly[idx] = value;    // ❌ Compile error
    
    // Read-write access
    readwrite[idx] = value;      // ✅ Valid
}
```

## Backend Memory Abstractions

### 1. Trait-Based Backend Selection

Each backend implements memory operations through specialized traits:

```cpp
namespace alpaka::onHost::internal {
    // Device memory allocation trait
    template<typename T_Type, typename T_Device, typename T_Extents>
    struct Alloc::Op<T_Type, T_Device, T_Extents> {
        auto operator()(T_Device& device, T_Extents const& extents) const;
    };
    
    // Unified memory allocation trait  
    template<typename T_Type, typename T_Device, typename T_Extents>
    struct AllocUnified::Op<T_Type, T_Device, T_Extents> {
        auto operator()(T_Device& device, T_Extents const& extents) const;
    };
}
```

### 2. Memory Transfer Operations

Alpaka provides unified memory transfer operations:

```cpp
// Synchronous memory copy
onHost::memcpy(queue, dst, src);

// Asynchronous memory copy  
onHost::memcpy(queue, dst, src);
onHost::wait(queue);  // Explicit synchronization

// Memory initialization
onHost::memset(queue, buffer, 0x00);  // Set to zero
```

### 3. Cross-Backend Memory Compatibility

```cpp
// Automatic backend detection and optimal transfer paths
auto hostBuffer = onHost::allocHost<float>(extents);
auto cudaBuffer = onHost::alloc<float>(cudaDevice, extents);
auto hipBuffer = onHost::alloc<float>(hipDevice, extents);

// Alpaka handles backend-specific transfer optimizations
onHost::memcpy(queue, cudaBuffer, hostBuffer);  // Host→CUDA optimized
onHost::memcpy(queue, hipBuffer, hostBuffer);   // Host→HIP optimized  
```

## Advanced Memory Features

### 1. Memory Prefetching and Hints

```cpp
// Unified memory prefetching (for backends that support it)
onHost::prefetch(queue, unifiedBuffer, targetDevice);

// Memory access hints
onHost::advise(queue, unifiedBuffer, advice::PreferredLocation, device);
```

### 2. Memory Pool Management

```cpp
// Memory pool allocation for reduced allocation overhead
auto memoryPool = onHost::makeMemoryPool(device);
auto buffer = memoryPool.alloc<float>(extents);
```

### 3. Memory Alignment Queries

```cpp
// Query optimal alignment for data type and backend
constexpr auto alignment = getOptimalAlignment<float>(api::cuda, deviceKind::nvidiaGpu);

// Create aligned views
auto alignedView = makeView(api, ptr, extents, Alignment<alignment>{});
```

## Performance Considerations

### 1. Memory Access Patterns

**Optimal Patterns:**
- **Coalesced access**: Consecutive threads access consecutive memory locations
- **Aligned access**: Memory addresses are properly aligned for vector operations
- **Pitched memory**: Use proper pitch calculations for 2D+ arrays

**Anti-Patterns:**
- **Strided access**: Large gaps between accessed elements
- **Random access**: Unpredictable memory access patterns
- **Frequent host-device transfers**: Minimize data movement

### 2. Memory Type Selection Guide

```cpp
// High-performance computation with minimal host interaction
auto deviceMem = onHost::alloc<float>(device, extents);

// Frequent host-device data exchange
auto unifiedMem = onHost::allocUnified<float>(device, extents);

// Host preprocessing with device acceleration  
auto mappedMem = onHost::allocMapped<float>(device, extents);

// Temporary computation data
auto deferredMem = onHost::allocDeferred<float>(queue, extents);
```

### 3. Shared Memory Optimization

```cpp
// Optimal shared memory usage
ALPAKA_FN_ACC void optimizedKernel(auto const& acc, concepts::MdSpan auto data) {
    // Size shared memory to match workgroup dimensions
    auto blockSize = getExtentsOf<Block>(acc);
    auto sharedMem = declareSharedMdArray<float, 0>(acc, blockSize);
    
    // Cooperative loading: all threads load data
    auto threadIdx = getIdxWithin<Block>(acc);
    auto globalIdx = getIdxWithin<Grid>(acc);
    
    sharedMem[threadIdx] = data[globalIdx];
    onAcc::syncBlockThreads(acc);
    
    // Process shared data (much faster than global memory)
    auto result = processSharedData(sharedMem, threadIdx);
    onAcc::syncBlockThreads(acc);
    
    data[globalIdx] = result;
}
```

## Integration with Alpaka Execution Model

### 1. Memory in Kernel Launch

```cpp
// Memory allocation
auto inputA = onHost::alloc<float>(device, extents);
auto inputB = onHost::alloc<float>(device, extents);  
auto output = onHost::alloc<float>(device, extents);

// Create views for kernel parameters
auto viewA = makeView(inputA);
auto viewB = makeView(inputB);
auto viewC = makeView(output);

// Kernel launch with memory views
queue.enqueue(executor, frameSpec, KernelBundle{
    VectorAddKernel{}, viewA, viewB, viewC, extents
});
```

### 2. Memory Lifetime Management

```cpp
{
    auto buffer = onHost::alloc<float>(device, extents);
    
    // Ensure buffer lifetime extends beyond kernel execution
    buffer.destructorWaitFor(queue);
    
    queue.enqueue(executor, frameSpec, KernelBundle{MyKernel{}, buffer});
    
    // buffer automatically freed after queue operations complete
}
```

## Summary

Alpaka's memory system provides a **unified, type-safe, and performance-oriented** abstraction for memory management across heterogeneous computing platforms. Key strengths include:

1. **Backend Agnostic**: Single API works across CPU, CUDA, HIP, and OneAPI
2. **Type Safety**: Compile-time guarantees for memory access patterns
3. **Performance**: Backend-specific optimizations while maintaining portability
4. **Flexibility**: Multiple memory types for different use cases
5. **RAII**: Automatic memory management with deterministic cleanup
6. **View System**: Efficient non-owning access to multi-dimensional data

The system successfully abstracts the complexity of heterogeneous memory hierarchies while providing the performance and control needed for high-performance computing applications.