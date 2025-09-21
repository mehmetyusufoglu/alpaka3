# Alpaka Practical Usage Functions Guide

## Parallel Algorithms

This document provides a comprehensive overview of Alpaka's parallel algorithms and iterator functions, their usage contexts, and implementation details.

### Host-Side Parallel Algorithms

| **Function** | **Namespace** | **Host Callable** | **Kernel Callable** | **Parameters** | **Purpose** |
|--------------|---------------|------------------|---------------------|----------------|-------------|
| **`reduce`** | `alpaka::onHost` | ✅ Yes | ❌ No | queue, executor, neutralElement, output, reduceFn, input | Parallel reduction operation |
| **`transformReduce`** | `alpaka::onHost` | ✅ Yes | ❌ No | queue, executor, neutralElement, output, reduceFn, transformFn, inputs... | Transform then reduce operation |
| **`transform`** | `alpaka::onHost` | ✅ Yes | ❌ No | queue, executor, output, transformFn, inputs... | Element-wise transformation |
| **`iota`** | `alpaka::onHost` | ✅ Yes | ❌ No | queue, executor, initValue, outputs... | Fill with sequential values |
| **`fill`** | `alpaka::onHost` | ✅ Yes | ❌ No | queue, dest, value, extents | Fill with constant value |

### Loop and Iteration Functions

| **Function** | **Namespace** | **Host Callable** | **Kernel Callable** | **Parameters** | **Purpose** |
|--------------|---------------|------------------|---------------------|----------------|-------------|
| **`ndLoopIncIdx`** | `alpaka::meta` | ✅ Yes | ✅ Yes | extent, function | Multi-dimensional nested loop |
| **`makeIdxMap`** | `alpaka::onAcc` | ❌ No | ✅ Yes | acc, workGroup, range, [traverse], [layout] | Create iterator container for kernels |

## Detailed Function Analysis

### 1. Reduction Operations

#### `alpaka::onHost::reduce`
```cpp
// Namespace: alpaka::onHost
template<typename DataType, typename T_Device, queueKind::concepts::QueueKind T_QueueKind>
inline void reduce(
    Queue<T_Device, T_QueueKind> const& queue,
    alpaka::concepts::Executor auto const exec,
    DataType const& neutralElement,
    alpaka::concepts::MdSpan auto out,
    auto&& binaryReduceFn,
    auto&& in)
```

**Usage Context:**
- **Host Only**: Called from host code to launch reduction kernels
- **Asynchronous**: Executes on device through queue
- **Multi-dimensional**: Supports n-dimensional data
- **Backend Agnostic**: Works with all Alpaka backends

**Example:**
```cpp
// Reduce operation
auto neutralElement = 0.0f;
onHost::reduce(queue, exec::cpuSerial, neutralElement, resultView, std::plus<>{}, inputView);
```

#### `alpaka::onHost::transformReduce`
```cpp
// Namespace: alpaka::onHost  
template<typename DataType, typename T_Device, queueKind::concepts::QueueKind T_QueueKind>
inline void transformReduce(
    Queue<T_Device, T_QueueKind> const& queue,
    alpaka::concepts::Executor auto const exec,
    DataType const& neutralElement,
    alpaka::concepts::MdSpan auto out,
    auto&& binaryReduceFn,
    auto&& transformFn,
    auto&&... in)
```

**Usage Context:**
- **Transform-Reduce Pattern**: Applies transformation before reduction
- **Multiple Inputs**: Supports variadic input arguments
- **SIMD Optimized**: Internal implementation uses SIMD operations
- **Atomic Operations**: Uses atomic operations for final reduction

**Example:**
```cpp
// Dot product implementation
auto dotProduct = [](auto a, auto b) { return a * b; };
onHost::transformReduce(queue, exec, 0.0f, result, std::plus<>{}, dotProduct, vectorA, vectorB);
```

### 2. Transformation Operations

#### `alpaka::onHost::transform`
```cpp
// Namespace: alpaka::onHost
template<typename T_Device, queueKind::concepts::QueueKind T_QueueKind>
inline void transform(
    Queue<T_Device, T_QueueKind> const& queue,
    alpaka::concepts::Executor auto const exec,
    auto&& out,
    auto&& fn,
    auto&&... in)
```

**Usage Context:**
- **Element-wise Operations**: Applies function to each element
- **Multiple Inputs**: Supports multiple input arrays
- **In-place or Out-of-place**: Can transform in-place or to different output
- **Kernel-based**: Launches SIMD-optimized kernels

**Example:**
```cpp
// Element-wise addition
auto addFunc = [](auto a, auto b) { return a + b; };
onHost::transform(queue, exec, outputView, addFunc, inputViewA, inputViewB);
```

#### `alpaka::onHost::iota`
```cpp
// Namespace: alpaka::onHost
template<typename T_DataType, typename T_Device, queueKind::concepts::QueueKind T_QueueKind>
inline void iota(
    Queue<T_Device, T_QueueKind> const& queue,
    alpaka::concepts::Executor auto const exec,
    T_DataType const& initValue,
    auto&& out0,
    auto&&... outOther)
```

**Usage Context:**
- **Sequential Fill**: Fills data with incrementing sequence
- **Multi-dimensional**: Increments fastest in last dimension
- **Multiple Outputs**: Can fill multiple arrays simultaneously
- **Fundamental Types Only**: Restricted to basic numeric types

**Example:**
```cpp
// Fill array with sequence 0, 1, 2, 3, ...
onHost::iota<int>(queue, exec, 0, arrayView);
```

### 3. Loop and Iteration Functions

#### `alpaka::meta::ndLoopIncIdx`
```cpp
// Namespace: alpaka::meta
template<typename TExtentVec, typename TFnObj>
auto ndLoopIncIdx(TExtentVec const& extent, TFnObj const& f) -> void
```

**Usage Context:**
- **Host and Kernel**: Can be called from both host and device code
- **Multi-dimensional Loops**: Nested loops over n-dimensional ranges
- **Index-based**: Provides current index to callback function
- **Compile-time Unrolling**: Optimized nested loop generation

**Example:**
```cpp
// Host-side usage
auto extent = Vec{width, height, depth};
meta::ndLoopIncIdx(extent, [&](auto idx) {
    // Process element at index idx
    data[idx] = computeValue(idx);
});

// Kernel-side usage
class MyKernel {
    ALPAKA_FN_ACC auto operator()(auto const& acc, auto extent, auto data) const -> void {
        meta::ndLoopIncIdx(extent, [&](auto idx) {
            data[idx] = processElement(idx);
        });
    }
};
```

#### `alpaka::onAcc::makeIdxMap`
```cpp
// Namespace: alpaka::onAcc
template<concepts::IdxTraversing T_Traverse = traverse::Flat, 
         concepts::IdxMapping T_IdxLayout = layout::Optimized>
ALPAKA_FN_HOST_ACC constexpr auto makeIdxMap(
    auto const& acc,
    auto const workGroup,
    auto const range,
    T_Traverse traverse = T_Traverse{},
    T_IdxLayout idxLayout = T_IdxLayout{})
```

**Usage Context:**
- **Kernel Only**: Can only be called from within kernels
- **Thread Mapping**: Maps threads to index ranges efficiently
- **Range-based Loops**: Returns container for range-based for loops
- **Traversal Policies**: Supports different iteration patterns

**Example:**
```cpp
class IteratorKernel {
    ALPAKA_FN_ACC auto operator()(auto const& acc, auto inputView) const -> void {
        // Basic iteration over entire range
        for(auto idx : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, IdxRange{inputView.getExtents()})) {
            inputView[idx] = computeValue(idx);
        }
        
        // With boundary conditions
        auto boundaries = BoundaryDirection{...};
        for(auto idx : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, IdxRange{inputView.getExtents()}, boundaries)) {
            inputView[idx] = processBoundary(idx);
        }
    }
};
```

## Iterator Types and Their Features

### Iterator Functionality Matrix

| **Iterator Type** | **Namespace** | **Multi-Dimensional** | **Host Usage** | **Kernel Usage** | **Boundary Support** | **Memory Pattern** |
|-------------------|---------------|----------------------|----------------|------------------|---------------------|-------------------|
| **`makeIdxMap`** | `alpaka::onAcc` | ✅ Yes | ❌ No | ✅ Yes | ✅ Yes | Thread-optimized |
| **`MdForwardIter`** | `alpaka` | ✅ Yes | ✅ Yes | ✅ Yes | ❌ No | Sequential traversal |
| **`BoundaryIter`** | `alpaka` | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes | Boundary-aware |
| **`FlatIdxContainer`** | `alpaka::onAcc` | ✅ Yes | ❌ No | ✅ Yes | ❌ No | Flat memory layout |
| **`TiledIdxContainer`** | `alpaka::onAcc` | ✅ Yes | ❌ No | ✅ Yes | ❌ No | Tiled memory access |

### 1. Index Mapping Iterators (`makeIdxMap`)

#### Basic Index Iteration
```cpp
// In kernel code - basic iteration
for(auto idx : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, IdxRange{extents})) {
    data[idx] = processElement(idx);
}
```

#### Traversal Policies
```cpp
// Flat traversal (default)
auto flatIter = onAcc::makeIdxMap(acc, workGroup, range, traverse::Flat{});

// Tiled traversal for better memory locality
auto tiledIter = onAcc::makeIdxMap(acc, workGroup, range, traverse::Tiled{});
```

#### Layout Mapping Policies
```cpp
// Optimized layout (default) - best performance for backend
auto optIter = onAcc::makeIdxMap(acc, workGroup, range, traverse::Flat{}, layout::Optimized{});

// Strided layout - regular stride pattern
auto stridedIter = onAcc::makeIdxMap(acc, workGroup, range, traverse::Flat{}, layout::Strided{});

// Contiguous layout - contiguous memory access
auto contiguousIter = onAcc::makeIdxMap(acc, workGroup, range, traverse::Flat{}, layout::Contiguous{});
```

### 2. Multi-Dimensional Forward Iterator

```cpp
// Host or kernel usage
auto mdIter = MdForwardIter{mdSpanView};
for(auto it = mdIter; it != mdIter.end(); ++it) {
    *it = processElement();
}

// Range-based loop (C++20)
for(auto& element : mdSpanView) {
    element = processElement();
}
```

### 3. Boundary-Aware Iteration

```cpp
// Boundary iterator with halo regions
auto boundaries = BoundaryDirection{
    Vec{left, right},     // X-direction boundaries
    Vec{bottom, top},     // Y-direction boundaries  
    Vec{front, back}      // Z-direction boundaries
};

// Kernel usage with boundary iterator
class BoundaryKernel {
    ALPAKA_FN_ACC auto operator()(auto const& acc, auto view, auto boundaries) const -> void {
        for(auto idx : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, IdxRange{view.getExtents()}, boundaries)) {
            // Automatically handles boundary conditions
            view[idx] = processBoundaryElement(idx);
        }
    }
};
```

### 4. Work Group Specifications

#### Thread Groups for `makeIdxMap`
```cpp
// All threads in grid
onAcc::worker::threadsInGrid

// Threads in current block
onAcc::worker::threadsInBlock  

// Linearized threads in block
onAcc::worker::linearThreadsInBlock

// All blocks in grid
onAcc::worker::blocksInGrid
```

## Performance Characteristics

### Algorithm Performance Matrix

| **Algorithm** | **Memory Pattern** | **CPU Performance** | **GPU Performance** | **Scalability** | **Best Use Case** |
|---------------|-------------------|--------------------|--------------------|-----------------|-------------------|
| **reduce** | Irregular (reduction tree) | Good | Excellent | High | Sum, min, max operations |
| **transformReduce** | Irregular | Good | Excellent | High | Dot products, norms |
| **transform** | Regular | Excellent | Excellent | High | Element-wise operations |
| **iota** | Sequential write | Good | Good | Medium | Data initialization |
| **makeIdxMap** | Depends on policy | Good | Excellent | High | Custom kernel loops |

### Iterator Performance Guidelines

#### Memory Access Optimization
```cpp
// Best: Contiguous access pattern for CPU
auto contiguousIter = onAcc::makeIdxMap(acc, workGroup, range, traverse::Flat{}, layout::Contiguous{});

// Best: Optimized layout for GPU (coalesced access)
auto optimizedIter = onAcc::makeIdxMap(acc, workGroup, range, traverse::Flat{}, layout::Optimized{});

// Special: Tiled access for cache optimization
auto tiledIter = onAcc::makeIdxMap(acc, workGroup, range, traverse::Tiled{});
```

#### Work Distribution Strategies
```cpp
// Fine-grained: Each thread processes few elements
for(auto idx : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, range)) {
    // Process single element
}

// Coarse-grained: Each thread processes multiple elements
for(auto idx : onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, range)) {
    // Process block of elements
}
```

## Usage Patterns and Best Practices

### 1. Algorithm Selection Guidelines

#### Choose `reduce` when:
- Simple aggregation operations (sum, product, min, max)
- Input data doesn't need transformation
- Binary associative operations

#### Choose `transformReduce` when:
- Need to transform data before reduction
- Complex operations like dot products, norms
- Multiple input arrays

#### Choose `transform` when:
- Element-wise operations on arrays
- Data conversion or filtering
- No reduction needed

### 2. Iterator Selection Guidelines

#### Use `makeIdxMap` when:
- Inside kernels for custom loops
- Need thread-optimized iteration
- Working with complex access patterns

#### Use `MdForwardIter` when:
- Host-side iteration over multi-dimensional data
- Sequential processing required
- Simple forward iteration

#### Use boundary iterators when:
- Working with stencil computations
- Need halo region handling
- Edge case processing

### 3. Performance Optimization Tips

#### Memory Access Optimization
```cpp
// Good: Align data access with memory layout
auto layout = layout::Contiguous{};  // For CPU
auto layout = layout::Optimized{};   // For GPU

// Good: Use appropriate traversal for algorithm
auto traverse = traverse::Tiled{};   // For cache-friendly access
auto traverse = traverse::Flat{};    // For simple linear access
```

#### Thread Mapping Optimization
```cpp
// CPU: Favor block-level iteration
auto iter = onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, range);

// GPU: Favor grid-level iteration  
auto iter = onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, range);
```

## Common Usage Examples

### Example 1: Vector Addition with Transform
```cpp
// Host side
auto addKernel = [](auto a, auto b) { return a + b; };
onHost::transform(queue, exec, resultView, addKernel, vectorA, vectorB);
```

### Example 2: Dot Product with TransformReduce
```cpp
// Host side
auto multiply = [](auto a, auto b) { return a * b; };
onHost::transformReduce(queue, exec, 0.0f, result, std::plus<>{}, multiply, vectorA, vectorB);
```

### Example 3: Custom Stencil Kernel with Iterator
```cpp
class StencilKernel {
    ALPAKA_FN_ACC auto operator()(auto const& acc, auto input, auto output) const -> void {
        for(auto idx : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, IdxRange{output.getExtents()})) {
            // 5-point stencil computation
            auto result = input[idx];
            if(idx.x() > 0) result += input[idx - Vec{1,0}];
            if(idx.x() < input.getExtents().x()-1) result += input[idx + Vec{1,0}];
            if(idx.y() > 0) result += input[idx - Vec{0,1}];
            if(idx.y() < input.getExtents().y()-1) result += input[idx + Vec{0,1}];
            output[idx] = result / 5.0f;
        }
    }
};
```

### Example 4: N-Dimensional Loop Processing
```cpp
// Host-side validation
auto extents = Vec{width, height, depth};
meta::ndLoopIncIdx(extents, [&](auto idx) {
    auto expected = computeExpectedValue(idx);
    auto actual = resultData[idx];
    assert(std::abs(expected - actual) < tolerance);
});
```

This comprehensive guide covers all major parallel algorithms and iterator functions in Alpaka, their appropriate usage contexts, and performance optimization strategies for different hardware backends.