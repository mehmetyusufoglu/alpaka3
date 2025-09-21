# Alpaka Basic Implementation Guide

## Complete Implementation Matrix

### API-Device-Executor Relationships

The three-layer model enables flexible hardware mapping:

| **Executor** | **Implements API** | **Executes On Device** | **Programming Model** | **Use Case** |
|-------------|-------------------|----------------------|---------------------|--------------|
| `exec::CpuSerial` | `api::host` | `deviceKind::cpu` | Sequential C++ | Debugging, small workloads |
| `exec::CpuOmpBlocks` | `api::host` | `deviceKind::cpu` | OpenMP parallel | Multi-core CPU workloads |
| `exec::GpuCuda` | `api::cuda` | `deviceKind::nvidiaGpu` | CUDA parallel | High-performance NVIDIA GPU |
| `exec::GpuHip` | `api::hip` | `deviceKind::amdGpu` | HIP parallel | High-performance AMD GPU |
| `exec::OneApi` | `api::oneApi` | `deviceKind::intelGpu` | SYCL parallel | Intel GPU optimization |
| `exec::OneApi` | `api::oneApi` | `deviceKind::cpu` | SYCL parallel | Intel CPU with SYCL backend |

### Implementation Characteristics

| **Backend** | **Memory Model** | **Synchronization** | **Compilation** | **Performance** |
|-------------|------------------|--------------------|-----------------|-----------------| 
| **CPU Serial** | Shared memory | No sync needed | Standard C++ | Low (single thread) |
| **CPU OpenMP** | Shared memory | OpenMP barriers | OpenMP-enabled compiler | Medium (multi-core) |
| **CUDA** | Device + Host memory | CUDA sync primitives | NVCC compiler | High (massively parallel) |
| **HIP** | Device + Host memory | HIP sync primitives | HIP compiler | High (massively parallel) |
| **OneAPI** | Unified memory | SYCL barriers | DPC++ compiler | High (adaptive) |

This design allows:
- **Same API on different hardware**: OneAPI can target both Intel GPUs and CPUs
- **Different execution strategies**: Host API can use serial or OpenMP execution  
- **Backend flexibility**: Multiple ways to utilize the same hardware
- **Future extensibility**: New combinations can be added without API changes

## Basic Data Structures and Their Features

### Memory and Data Management Structures

| **Structure** | **Multi-Dimensional** | **OnHost Support** | **OnAcc Support** | **Pass Kernel as Parameter?** | **Ownership** | **Memory Location** | **Primary Use Case** |
|---------------|----------------------|-------------------|-------------------|---------------------|---------------|--------------------|--------------------|
| **Buffer** | ✅ Yes | ✅ Yes | ❌ No | ❌ No | 🔒 Owning | Device/Host | Memory allocation |
| **View** | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes | 📖 Non-owning | Any | Kernel data access |
| **MdSpan** | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes | 📖 Non-owning | Any | Standard C++ interface |
| **Iter** | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes | 📖 Non-owning | Any | Element iteration |
| **BoundaryIter** | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes | 📖 Non-owning | Any | Boundary-aware iteration |
| **SubView** | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes | 📖 Non-owning | Any | Subset access |

### Detailed Structure Analysis

#### Buffer - Memory Allocation
```cpp
// Creation (OnHost only)
auto buffer = onHost::allocAsyncBuffer<float, 2>(device, {width, height});

// Characteristics:
// ✅ Multi-dimensional: Support 1D, 2D, 3D, etc.
// ✅ OnHost: Created and managed on host
// ❌ OnAcc: Cannot be created inside kernels
// ❌ Kernel Parameter: Not passed directly to kernels
// 🔒 Owning: Manages memory lifetime (RAII)
// 📍 Memory: Device or host memory depending on device type
```

#### View - Data Access Interface  
```cpp
// Creation from buffer
auto view = makeView(buffer);

// Kernel usage
class MyKernel {
    ALPAKA_FN_ACC auto operator()(auto const& acc, auto const& view) const -> void {
        auto idx = acc.getIdxWithin();
        view[idx] = computeValue(idx);  // Direct element access
    }
};

// Characteristics:
// ✅ Multi-dimensional: Preserves buffer dimensionality
// ✅ OnHost: Can be created and used on host
// ✅ OnAcc: Fully usable inside kernels
// ✅ Kernel Parameter: Primary way to pass data to kernels
// 📖 Non-owning: References existing memory
// 📍 Memory: Points to buffer's memory location
```

#### MdSpan - Standard C++ Multi-Dimensional Interface
```cpp
// Creation and kernel usage
class MyKernel {
    ALPAKA_FN_ACC auto operator()(auto const& acc, 
                                  alpaka::concepts::MdSpan auto const& data) const -> void {
        auto [i, j] = acc.getIdxWithin();
        data[i, j] = computeValue(i, j);  // Standard C++23 syntax
    }
};

// Characteristics:
// ✅ Multi-dimensional: Based on C++23 std::mdspan
// ✅ OnHost: Full support for host operations
// ✅ OnAcc: Optimized for kernel usage
// ✅ Kernel Parameter: Preferred modern interface
// 📖 Non-owning: View into existing memory
// 📍 Memory: Backend-agnostic memory access
```

#### Iterator Types - Element Access Patterns
```cpp
// Basic iterator usage in kernels
class IteratorKernel {
    ALPAKA_FN_ACC auto operator()(auto const& acc, auto const& iter) const -> void {
        auto myIter = iter + acc.getIdxWithin();  // Offset by thread index
        *myIter = computeValue();  // Direct element access
    }
};

// Boundary iterator for edge handling
class BoundaryKernel {
    ALPAKA_FN_ACC auto operator()(auto const& acc, auto const& boundaryIter) const -> void {
        auto idx = acc.getIdxWithin();
        // Automatically handles boundary conditions
        auto value = boundaryIter[idx];  // Safe access even at edges
    }
};
```

### Usage Patterns and Best Practices

#### 1. Memory Allocation Pattern
```cpp
// Host-side resource management
auto device = makeDevice(deviceSpec);
auto queue = device.makeQueue();

// Allocate owning buffer
auto buffer = onHost::allocAsyncBuffer<DataType, Dims>(device, extents);

// Create non-owning view for kernel access
auto view = makeView(buffer);
```

#### 2. Kernel Parameter Patterns
```cpp
// Pattern 1: View-based (Traditional Alpaka)
class ViewKernel {
    ALPAKA_FN_ACC auto operator()(auto const& acc, auto const& view) const -> void;
};

// Pattern 2: MdSpan-based (Modern C++)  
class MdSpanKernel {
    ALPAKA_FN_ACC auto operator()(auto const& acc, 
                                  alpaka::concepts::MdSpan auto const& data) const -> void;
};

// Pattern 3: Iterator-based (Algorithm-style)
class IteratorKernel {
    ALPAKA_FN_ACC auto operator()(auto const& acc, auto const& iter) const -> void;
};
```

#### 3. Multi-Dimensional Access Patterns
```cpp
// 2D access with View
view[{i, j}] = value;                    // Index-based access
view[acc.getIdxWithin()] = value;        // Thread-index access

// 2D access with MdSpan  
data[i, j] = value;                      // C++23 standard syntax
data[linearIndex] = value;               // Linear indexing

// Iterator arithmetic
auto element = iter + offset;            // Pointer arithmetic style
auto value = boundaryIter[idx];          // Safe boundary access
```

### Memory Management Lifecycle

#### Resource Creation and Cleanup
```cpp
// 1. Device Selection
auto device = onHost::makeDevice(deviceSpec);
auto queue = device.makeQueue();

// 2. Memory Allocation (Owning)
auto buffer = onHost::allocAsyncBuffer<float, 2>(device, {width, height});
//   ↑ RAII: Automatically freed when buffer goes out of scope

// 3. View Creation (Non-owning)  
auto view = makeView(buffer);
//   ↑ Safe: View lifetime must not exceed buffer lifetime

// 4. Kernel Execution
queue.enqueue(executor, frameSpec, MyKernel{}, view);
//   ↑ View passed by value/reference, no ownership transfer

// 5. Synchronization
queue.wait();
//   ↑ Ensures kernel completion before buffer destruction
```

#### Ownership Transfer and Lifetime Rules
```cpp
// ✅ Safe: Buffer owns memory, view references it
{
    auto buffer = allocAsyncBuffer<float, 1>(device, {1000});
    auto view = makeView(buffer);
    // Both buffer and view are valid here
} // ✅ Buffer destructor frees memory, view becomes invalid

// ❌ Dangerous: View outlives buffer
auto createDanglingView() {
    auto buffer = allocAsyncBuffer<float, 1>(device, {1000});
    return makeView(buffer);  // ❌ Buffer destroyed, view becomes dangling
}

// ✅ Safe: Return owning buffer, create view from it
auto createSafeBuffer() {
    return allocAsyncBuffer<float, 1>(device, {1000});  // ✅ Ownership transferred
}
```

### Performance Considerations

#### Memory Access Patterns
| **Structure** | **Access Pattern** | **Performance** | **Use Case** |
|---------------|-------------------|-----------------|--------------|
| **View** | Direct indexing | High | General purpose kernel access |
| **MdSpan** | Standard syntax | High | Modern C++ compatibility |
| **Iter** | Pointer arithmetic | Highest | Performance-critical loops |
| **BoundaryIter** | Safe boundary access | Medium-High | Edge case handling |

#### Compilation and Backend Optimization
- **Compile-time dispatch**: Structure choice affects template instantiation
- **Backend specialization**: Each backend optimizes access patterns differently
- **Memory coalescing**: GPU backends optimize contiguous access patterns
- **Vectorization**: CPU backends benefit from iterator-based patterns

This design provides:
- **Flexibility**: Multiple ways to access the same data
- **Safety**: Non-owning views prevent memory leaks
- **Performance**: Zero-cost abstractions with backend optimization
- **Standards compliance**: C++23 MdSpan compatibility for future-proofing