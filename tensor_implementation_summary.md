# Alpaka3 Tensor Library Implementation

## Overview
This implementation provides a foundation for a portable AI inference tensor library built on top of alpaka3. The library follows the patterns demonstrated in alpaka3 tutorials and examples like heatEquation2D and grayScale.

## File Structure
```
include/alpaka/tensor/
├── Tensor.hpp              # Original complex implementation (has compilation issues)
├── TensorSimple.hpp        # Simplified working tensor class
├── ops/
│   └── Elementwise.hpp     # Basic elementwise operations
└── tensor.hpp              # Main include header

example/tensor/
├── CMakeLists.txt          # Build configuration
└── src/
    └── tensorExample.cpp   # Demonstration example
```

## Core Components

### 1. Tensor Class (`TensorSimple.hpp`)
- **Template parameters**: `<typename T, std::size_t Rank>`
- **Key features**:
  - Shape management using `alpaka::Vec<std::size_t, Rank>`
  - Host storage with `std::vector<T>`
  - Device memory management with lazy allocation
  - DataType and Layout enums for AI-specific metadata
  - Memory transfer operations (toDevice/toHost)
  - Element access with bounds checking
  - Copy/move semantics

### 2. Operations (`ops/Elementwise.hpp`)
- **ElementwiseAddKernel**: Kernel for tensor addition
- **ElementwiseReluKernel**: Kernel for ReLU activation
- **High-level functions**: `add()`, `relu()`, `relu_inplace()`
- **Note**: Current implementation has placeholders for proper MdSpan integration

### 3. Example Application (`tensorExample.cpp`)
- Demonstrates tensor creation, memory management, and basic operations
- Tests compatibility checking, utilities, and copy operations
- Shows integration with alpaka3's device/queue system

## Key Design Decisions

### Memory Management
- **Host storage**: `std::vector<T>` (following tutorial patterns)
- **Device storage**: `std::shared_ptr<void>` wrapper around alpaka buffers
- **Lazy allocation**: Device memory allocated on demand
- **Transfer operations**: Async copies using alpaka queues

### AI-Specific Features
- **DataType enum**: Runtime type information (Float32, Float16, Int8, etc.)
- **Layout enum**: Memory layout optimization (RowMajor, NCHW, NHWC, etc.)
- **Metadata**: Name, device context, compatibility checking
- **Shape semantics**: Beyond just dimensions, understanding tensor structure

### Alpaka3 Integration
- Uses `alpaka::Vec` for shape representation
- Follows `onHost::` namespace patterns for host-side operations
- Integrates with alpaka's device/queue system
- Prepared for kernel integration via MdSpan views

## Current Status

### ✅ Implemented
- Core tensor class with shape, storage, and metadata
- Basic memory management (allocation, deallocation)
- Host-side operations (fill, zero, element access)
- Copy/move semantics
- Compatibility checking
- Basic operation structure (kernels defined)
- Working example application

### 🚧 Partial Implementation
- Device memory transfers (placeholders exist)
- Kernel launching (structure in place, needs MdSpan integration)
- MdSpan view creation (needs proper device buffer handling)

### ❌ Not Yet Implemented
- Full MdSpan integration with device buffers
- Complete kernel execution pipeline
- SIMD operations using `onAcc::SimdAlgo`
- Matrix operations (GEMM, convolution)
- Memory pooling and optimization
- Graph execution system
- Model loading (ONNX integration)

## Next Steps (Level 1 Completion)

1. **Fix MdSpan Integration**
   - Properly expose device buffers as MdSpan views
   - Implement actual memory transfers in toDevice/toHost
   - Complete kernel launching pipeline

2. **Add SIMD Operations**
   - Integrate `onAcc::SimdAlgo{onAcc::worker::threadsInGrid}` pattern
   - Implement vectorized elementwise operations
   - Follow grayScale example patterns

3. **Expand Operation Set**
   - Matrix multiplication (basic blocked implementation)
   - More activation functions (sigmoid, tanh)
   - Reduction operations (sum, mean)
   - Transpose and reshape operations

4. **Testing and Validation**
   - Unit tests for all operations
   - Numerical accuracy tests
   - Performance benchmarks
   - Multi-device testing

5. **Documentation**
   - API documentation
   - Usage examples
   - Performance guidelines

## Integration with Alpaka3 Examples

The tensor library follows patterns from:
- **heatEquation2D**: Multi-dimensional memory management, chunked execution
- **grayScale**: SIMD operations, kernel launching
- **Tutorial examples**: Memory allocation, device selection, queue management

## Compilation and Testing

To build and test the tensor library:

1. Ensure alpaka3 is properly configured with examples enabled
2. The tensor example should be automatically included in the build
3. Run the example to test basic functionality

```bash
cd build
make tensorExample
./example/tensor/tensorExample
```

This implementation provides a solid foundation for Level 1 of the tensor library development strategy, with clear paths for expansion to Level 2 (additional operations) and Level 3 (model import and code generation).
