# Alpaka Library Dependency Graph and Architecture Overview

This document provides a comprehensive analysis of the Alpaka library's architecture, dependency relationships, and hierarchical structure based on detailed code analysis.

## Architecture Overview

Alpaka follows a layered architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────┐
│                    User Application                     │
├─────────────────────────────────────────────────────────┤
│                  alpaka.hpp (Umbrella)                  │
├─────────────────────────────────────────────────────────┤
│     onHost/          │          onAcc/                 │
│  (Host Operations)   │   (Accelerator Operations)      │
├─────────────────────────────────────────────────────────┤
│              Three-Layer Abstraction Model              │
│  ┌─────────────┬─────────────────┬─────────────────────┐ │
│  │ Executors   │  Device Kinds   │      APIs           │ │
│  │ (Strategy)  │  (Hardware)     │  (Programming)      │ │
│  ├─────────────┼─────────────────┼─────────────────────┤ │
│  │cpuSerial    │ cpu             │ host                │ │
│  │ompBlocks    │ amdGpu          │ cuda                │ │
│  │gpuCuda      │ nvidiaGpu       │ hip                 │ │
│  │gpuHip       │ intelGpu        │ oneApi              │ │
│  │oneApi       │                 │                     │ │
│  └─────────────┴─────────────────┴─────────────────────┘ │
├─────────────────────────────────────────────────────────┤
│     Core Infrastructure │  Memory Management │ Math    │
├─────────────────────────────────────────────────────────┤
│              Backend-Specific Implementations           │
│  cuDNN │ cuBLAS │ rocBLAS │ MKL │ Standard Library     │
└─────────────────────────────────────────────────────────┘
```

## Three-Layer Abstraction Model

Alpaka's key innovation is the separation of concerns through a three-layer abstraction:

### 1. APIs (Programming Models)
Define how parallel code is written and compiled:
- **`api::host`**: Standard C++ with threading libraries
- **`api::cuda`**: NVIDIA CUDA programming model
- **`api::hip`**: AMD HIP programming model
- **`api::oneApi`**: Intel OneAPI/SYCL standard

### 2. Device Kinds (Hardware Categories)  
Classify computational hardware by capabilities:
- **`deviceKind::cpu`**: Multi-core processors with SIMD
- **`deviceKind::amdGpu`**: AMD graphics processors
- **`deviceKind::nvidiaGpu`**: NVIDIA graphics processors
- **`deviceKind::intelGpu`**: Intel graphics processors

### 3. Executors (Execution Strategies)
Combine APIs with specific parallelization approaches:
- **`exec::cpuSerial`**: Sequential execution on CPU
- **`exec::ompBlocks`**: OpenMP block-parallel execution
- **`exec::gpuCuda`**: CUDA GPU execution strategy
- **`exec::gpuHip`**: HIP GPU execution strategy
- **`exec::oneApi`**: OneAPI execution (CPU or GPU)

### Mapping Flexibility

This three-layer model enables sophisticated hardware mapping:

```
Executor        API         Device Kind     Example Use Case
────────────────────────────────────────────────────────────
cpuSerial   →   host    →   cpu         →   Debug/sequential
ompBlocks   →   host    →   cpu         →   Multi-core CPU
gpuCuda     →   cuda    →   nvidiaGpu   →   NVIDIA GPU
gpuHip      →   hip     →   amdGpu      →   AMD GPU
oneApi      →   oneApi  →   intelGpu    →   Intel GPU
oneApi      →   oneApi  →   cpu         →   Intel CPU with SYCL
```

### Accelerator Context Creation and Flow

The complete flow from Executor to Accelerator context:

```
Executor        makeAcc Function              Accelerator Context (Acc)        Kernel Parameter
─────────────────────────────────────────────────────────────────────────────────────────────
cpuSerial   →   makeAcc(exec::CpuSerial, ...)  →  Acc<cpu::Serial>         →   auto const& acc
ompBlocks   →   makeAcc(exec::CpuOmpBlocks, ...)→  Acc<cpu::OmpBlocks>      →   auto const& acc
gpuCuda     →   makeAcc(exec::GpuCuda, ...)    →  Acc<cuda::GpuStorage>    →   auto const& acc
gpuHip      →   makeAcc(exec::GpuHip, ...)     →  Acc<hip::GpuStorage>     →   auto const& acc
oneApi      →   makeAcc(exec::OneApi, ...)     →  Acc<sycl::DeviceStorage> →   auto const& acc
```

**Flow explanation**:
1. **Executor** - Runtime execution strategy selection
2. **makeAcc Function** - Compile-time dispatch to create accelerator context
3. **Accelerator Context** - Backend-specific `Acc<T_Storage>` wrapper
4. **Kernel Parameter** - The `acc` parameter passed to user kernel functions

**Key characteristics**:
- **Template Specialization**: Each executor creates a different `Acc<T_Storage>` type
- **Compile-Time Dispatch**: `makeAcc` uses function overloading for zero-overhead selection
- **Backend Storage**: `T_Storage` contains backend-specific data and functionality
- **Unified Interface**: All accelerator contexts provide the same user interface (`getIdxWithin`, `getExtentsOf`, etc.)

### makeAcc Function Parameters and Configuration

The `makeAcc` function requires specific parameters beyond just the executor:

```
Parameter Type      Description                    Example Values                   Purpose
─────────────────────────────────────────────────────────────────────────────────────────────
Executor           Runtime execution strategy     exec::cpuSerial                 Compile-time dispatch
                                                  exec::ompBlocks                 
                                                  exec::gpuCuda                    

ThreadBlocking     Thread hierarchy configuration ThreadSpec{numBlocks,           Defines parallelism structure
                                                            numThreadsPerBlock}   for hierarchical execution

NumBlocks          Number of blocks/work groups   Vec<size_t, 1>{64}              Block-level parallelism
                                                  Vec<size_t, 2>{8, 8}            (Grid dimension)

NumThreadsPerBlock Threads per block/work group   Vec<size_t, 1>{256}             Thread-level parallelism  
                                                  Vec<size_t, 2>{16, 16}          (Block dimension)
```

### Complete Parameter Flow from FrameSpec to makeAcc

```
Component            Source                          Extracted For makeAcc              Used In makeAcc
──────────────────────────────────────────────────────────────────────────────────────────────────────
FrameSpec           User-defined configuration      frameSpec.m_threadSpec             threadBlocking parameter
  └─ m_numFrames    Total work distribution         → m_numBlocks                      Block count configuration
  └─ m_frameExtent  Work per frame                  → (calculated internally)          Frame size for blocking
  └─ m_threadSpec   Thread hierarchy spec           → entire ThreadSpec                Direct parameter to makeAcc

ThreadSpec          From FrameSpec                  threadSpec.m_numBlocks             Block grid dimensions
  └─ m_numBlocks    Block/work group count          threadSpec.m_numThreads            Thread block dimensions
  └─ m_numThreads   Threads per block count
```

### Example Parameter Construction (from VectorAdd)

```cpp
// User configuration
Vec<size_t, 1u> chunkSize = 256u;
uint32_t elementsPerWorker = getNumElemPerThread<Data>(queue);

// FrameSpec creation
auto dataBlocking = onHost::FrameSpec{
    divCeil(extent, chunkSize * elementsPerWorker),  // numFrames (blocks)
    chunkSize                                        // frameExtent (threads per block)
};

// Internal makeAcc call (in queue.enqueue)
onAcc::Acc acc = makeAcc(executor, dataBlocking.m_threadSpec);
//                        ↑               ↑
//                    runtime          threadBlocking parameter
//                   executor          (contains numBlocks + numThreads)
```

## Hierarchical Parallelism Model

Alpaka implements a 5-level hierarchical parallelism abstraction:

### 1. Grid Level
- **Description**: The entire computational problem space
- **Purpose**: Defines the global work distribution
- **Synchronization**: Sequential within queue, parallel across queues
- **Backend Mapping**: 
  - CUDA: Grid of blocks
  - OpenMP: Parallel region
  - CPU: Thread pool

### 2. Block Level  
- **Description**: Independent groups of threads with shared resources
- **Purpose**: Enable local communication and shared memory
- **Synchronization**: No inter-block synchronization
- **Backend Mapping**:
  - CUDA: Thread block
  - OpenMP: Team
  - CPU: Work group

### 3. Warp Level
- **Description**: Groups of threads executing in lock-step (SIMD)
- **Purpose**: Vectorized execution and coalesced memory access
- **Synchronization**: Implicit synchronization within warp
- **Backend Mapping**:
  - CUDA: Warp (32 threads)
  - CPU: SIMD vector width
  - SYCL: Sub-group

### 4. Thread Level
- **Description**: Individual execution units within a block
- **Purpose**: Fine-grained parallelism with synchronization capabilities
- **Synchronization**: Block-local barriers and atomic operations
- **Backend Mapping**:
  - CUDA: Individual thread
  - OpenMP: Thread within team
  - CPU: Individual thread

### 5. Element Level
- **Description**: Data elements processed sequentially by each thread
- **Purpose**: Optimize register usage and enable vectorization
- **Synchronization**: None (sequential processing)
- **Backend Mapping**: Loop within thread execution

## Core Component Dependencies

### Foundation Layer (Core)

#### 1. Configuration System
- **`config.hpp`**: Central configuration management
  - Platform detection (OS, architecture)
  - Compiler feature detection
  - Backend availability detection
- **Dependencies**: None (foundation)
- **Used by**: All other components

#### 2. Type System and Metaprogramming
- **`Tag.hpp`**: Type-safe compile-time tags
- **`Dict.hpp`**: Typed key-value containers
- **`common.hpp`**: Platform-specific definitions
- **Dependencies**: Configuration system
- **Used by**: API layer, trait system

#### 3. Utility Infrastructure  
- **`Assert.hpp`**, **`Debug.hpp`**: Debugging support
- **`util.hpp`**: Common utilities
- **Dependencies**: Configuration, type system
- **Used by**: All components for debugging and utilities

### API Abstraction Layer

#### 1. Backend Detection and Selection
- **`api/api.hpp`**: Runtime API detection
  - `thisApi()`: Current execution context detection
  - Platform availability checking
- **Dependencies**: Core configuration, backend-specific headers
- **Used by**: Device management, queue creation

#### 2. Backend-Specific Implementations
- **`api/cuda/`**: NVIDIA CUDA backend
  - Device enumeration and management
  - Stream/event handling
  - Memory management
- **`api/hip/`**: AMD HIP backend
  - Similar structure to CUDA
  - Shared implementation via `unifiedCudaHip/`
- **`api/oneApi/`**: Intel OneAPI/SYCL backend
  - Queue and device management
  - SYCL-specific optimizations
- **`api/host/`**: CPU host backend
  - Thread pool management
  - OpenMP integration
  - Standard library algorithms

#### 3. Unified Interfaces
- **`api/unifiedCudaHip.hpp`**: Shared CUDA/HIP abstractions
- **`api/generic.hpp`**: Generic backend implementations
- **Dependencies**: Backend-specific implementations
- **Used by**: Higher-level abstractions

### Host Operations Layer (onHost)

#### 1. Device and Resource Management
- **`Device.hpp`**: Device abstraction and lifecycle
  - Template-based device selection
  - Backend-agnostic device interface
  - Resource handle management
- **`DeviceProperties.hpp`**: Device capability querying
- **`DeviceSelector.hpp`**: Automatic device selection strategies

#### 2. Execution Management
- **`Queue.hpp`**: Asynchronous execution management
  - Backend-specific queue implementations
  - Event-based synchronization
  - Task dependency management
- **`Event.hpp`**: Synchronization primitives
- **`ThreadSpec.hpp`**: Thread configuration

#### 3. Memory Management
- **`mem/`**: Host-side memory operations
  - Buffer allocation/deallocation
  - Memory transfers between host and device
  - Memory mapping and pinning

#### 4. Parallel Algorithms
- **`algo/`**: Host-side algorithm implementations
  - `reduce.hpp`: Parallel reductions
  - `transform.hpp`: Data transformations
  - `concurrent.hpp`: Concurrent algorithm utilities

### Accelerator Operations Layer (onAcc)

#### 1. Accelerator Context
- **`Acc.hpp`**: Main accelerator interface
  - Index and extent calculation methods
  - Thread hierarchy navigation
  - Backend-agnostic accelerator operations

#### 2. Memory Operations
- **`GlobalMem.hpp`**: Global memory access patterns
- **`layout.hpp`**: Memory layout abstractions
- **`memoryFence.hpp`**: Memory synchronization
- **`memoryScope.hpp`**: Memory scope definitions

#### 3. Synchronization and Atomics
- **`atomic.hpp`**: Cross-platform atomic operations
- **`atomicHierarchy.hpp`**: Hierarchical atomic operations
- **`WorkGroup.hpp`**: Block-level synchronization

#### 4. SIMD and Vectorization
- **`SimdAlgo.hpp`**: SIMD algorithm implementations
- **`internal/SimdConcurrent.hpp`**: Concurrent SIMD operations

### Memory Management Layer

#### 1. Memory Abstractions
- **`mem/View.hpp`**: Multi-dimensional data views
- **`mem/MdSpan.hpp`**: Multi-dimensional span interface
- **`mem/DataPitches.hpp`**: Stride and pitch calculations

#### 2. Access Patterns and Iterators
- **`mem/Iter.hpp`**: Iterator abstractions
- **`mem/BoundaryIter.hpp`**: Boundary-aware iteration
- **`mem/MdForwardIter.hpp`**: Multi-dimensional forward iteration

#### 3. Index Management
- **`mem/IdxRange.hpp`**: Index range utilities
- **`mem/ThreadSpace.hpp`**: Thread index space management

## Dependency Flow

### Compilation Dependencies
1. **Core** → **API** → **onHost/onAcc** → **User Code**
2. **Configuration** → **Platform Detection** → **Backend Selection**
3. **Type System** → **Trait System** → **Template Specializations**

### Runtime Dependencies
1. **Platform Detection** → **Backend Initialization** → **Device Enumeration**
2. **Device Selection** → **Queue Creation** → **Memory Allocation**
3. **Kernel Compilation** → **Execution** → **Synchronization**

### Memory Dependencies
1. **Host Memory** → **Device Memory** → **Kernel Execution**
2. **Buffer Management** → **Transfer Operations** → **Synchronization**

## Critical Design Patterns

### 1. Tag-Based Dispatch
```cpp
ALPAKA_TAG(deviceKind);
// Enables compile-time backend selection
```

### 2. Trait-Based Customization
```cpp
namespace trait {
    template<typename T_Backend>
    struct GetProperty {
        static auto getProperty(T_Backend const& backend);
    };
}
```

### 3. Handle-Based Resource Management
- RAII wrappers for all backend resources
- Automatic cleanup and exception safety
- Type-safe resource access

### 4. Concept-Based Interface Design
```cpp
template<alpaka::concepts::Api T_Api>
constexpr auto makeDevice(T_Api api);
```

## Integration and Extension Points

### Backend Integration
Each backend must implement:
1. **Device enumeration and properties**
2. **Queue/stream management**
3. **Memory allocation and transfer**
4. **Kernel compilation and execution**
5. **Synchronization primitives**

### User Extension Points
1. **Custom memory allocators**
2. **Custom synchronization strategies**
3. **Backend-specific optimizations**
4. **Domain-specific algorithm implementations**

This architecture enables Alpaka to provide performance portability while maintaining flexibility for future hardware platforms and programming models.
- **Core to Utility**: Utility components may depend on core components for logging and error reporting functionalities.
- **Accelerator-Specific to Core**: Accelerator-specific components often require core functionalities to manage resources and execute tasks efficiently.

## Visual Representation

A visual representation of the dependency graph will be created using PlantUML. This diagram will illustrate the components and their relationships, providing a clear overview of the architecture.

## Conclusion

The dependency graph serves as a valuable tool for understanding the structure and relationships within the Alpaka library. By analyzing these dependencies, developers can make informed decisions when modifying or extending the library, ensuring that changes do not inadvertently disrupt existing functionalities. 

Future updates to this document will include the actual dependency graph diagram once it has been generated.