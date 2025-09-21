# Alpaka Library Component Analysis

## Overview

This document provides a comprehensive analysis of the major components within the Alpaka portability library. Alpaka provides performance portability through a hierarchical abstraction of parallelism that unifies different accelerator backends (CPU, CUDA, HIP, SYCL).

## Architecture Principles

### Redundant Hierarchical Parallelism
Alpaka implements a 5-level hierarchy of parallelism:
1. **Grid** - The entire computational problem space
2. **Block** - Independent groups of threads with shared memory
3. **Warp** - Groups of threads executing in lock-step (SIMD)
4. **Thread** - Individual execution units
5. **Element** - Data elements processed by each thread

### Backend Abstraction
The library supports multiple backends through a unified API:
- **Host CPU** (OpenMP, std::thread)
- **NVIDIA CUDA**
- **AMD HIP** 
- **Intel OneAPI/SYCL**

## Major Component Categories

### 1. Core Infrastructure (`core/`)

#### Configuration and Platform Detection
- **`config.hpp`**: Compile-time configuration and feature detection
- **`common.hpp`**: Platform-specific macros and function attributes
- **`Cuda.hpp`, `Sycl.hpp`**: Platform-specific headers and utilities
- **`ApiCudaRt.hpp`, `ApiHipRt.hpp`**: Runtime API wrappers

#### Fundamental Types and Utilities
- **`Tag.hpp`**: Type-safe tagging system for compile-time dispatch
- **`Dict.hpp`**: Dictionary-like container for storing typed values
- **`Debug.hpp`, `Assert.hpp`**: Debugging and assertion utilities

### 2. API Abstraction Layer (`api/`)

#### Three-Layer Hardware Abstraction Model

Alpaka employs a sophisticated three-layer abstraction model:

##### APIs (Programming Models)
Programming model abstractions that define how code is written and executed:
- **`api::host`**: Host CPU programming model (C++, OpenMP, TBB)
- **`api::cuda`**: NVIDIA CUDA programming model
- **`api::hip`**: AMD HIP programming model  
- **`api::oneApi`**: Intel OneAPI/SYCL programming model

##### Device Kinds (Hardware Categories)
Physical hardware categories that define computational capabilities:
- **`deviceKind::cpu`**: CPU processors (multi-core, SIMD)
- **`deviceKind::amdGpu`**: AMD Graphics Processing Units
- **`deviceKind::nvidiaGpu`**: NVIDIA Graphics Processing Units
- **`deviceKind::intelGpu`**: Intel Graphics Processing Units

##### Executors (Execution Strategies)
Specific combinations of APIs and parallelization strategies:
- **`exec::cpuSerial`**: Sequential CPU execution
- **`exec::ompBlocks`**: OpenMP block-parallel CPU execution
- **`exec::gpuCuda`**: CUDA GPU execution strategy
- **`exec::gpuHip`**: HIP GPU execution strategy
- **`exec::oneApi`**: OneAPI/SYCL execution strategy

#### Backend Selection and Management
- **`api.hpp`**: Main API detection and runtime selection
  - `thisApi()`: Returns current execution context API
  - Support for multi-API environments
- **Backend-specific implementations**:
  - `cuda/`: NVIDIA CUDA backend
  - `hip/`: AMD HIP backend
  - `oneApi/`: Intel OneAPI/SYCL backend
  - `host/`: CPU host backend

#### Unified Interfaces
- **`unifiedCudaHip.hpp`**: Shared abstractions for CUDA/HIP
- **`generic.hpp`**: Generic backend implementations
- **`trait.hpp`**: Trait-based backend customization

#### API-Device-Executor Relationships
The three-layer model enables flexible hardware mapping:

```
API Layer        Device Kind       Executor
─────────────────────────────────────────────
api::host    →   deviceKind::cpu  →  exec::cpuSerial
api::host    →   deviceKind::cpu  →  exec::ompBlocks  
api::cuda    →   deviceKind::nvidiaGpu → exec::gpuCuda
api::hip     →   deviceKind::amdGpu    → exec::gpuHip
api::oneApi  →   deviceKind::intelGpu  → exec::oneApi
api::oneApi  →   deviceKind::cpu       → exec::oneApi
```

This design allows:
- **Same API on different hardware**: OneAPI can target both Intel GPUs and CPUs
- **Different execution strategies**: Host API can use serial or OpenMP execution
- **Backend flexibility**: Multiple ways to utilize the same hardware
- **Future extensibility**: New combinations can be added without API changes

#### Accelerator Context Hierarchy

```
Acc<T_Storage>
├── Executor (Execution Strategy)
└── FrameSpec (Thread Configuration)
    └── ThreadSpec
        ├── numBlocks (Grid Dimensions)
        └── numThreadsPerBlock (Block Size)
```

**1. Executor Types and Their Purpose:**

| **Executor** | **Target Hardware** | **Execution Model** | **Use Case** |
|--------------|--------------------|--------------------|--------------|
| `exec::cpuSerial` | CPU | Single-threaded | Sequential processing, debugging |
| `exec::ompBlocks` | CPU | Multi-threaded (OpenMP) | Parallel CPU workloads |
| `exec::gpuCuda` | NVIDIA GPU | Massively parallel | High-performance CUDA computing |
| `exec::gpuHip` | AMD GPU | Massively parallel | High-performance AMD GPU computing |
| `exec::oneApi` | Intel GPU/CPU | Unified parallel | Intel hardware optimization |

**2. FrameSpec Configuration Structure:**

| **Component** | **Type** | **Description** | **Hardware Impact** |
|---------------|----------|-----------------|-------------------|
| `ThreadSpec` | Container | Thread hierarchy configuration | Defines parallel execution layout |
| `numBlocks` | Dimensions | Grid size (how many blocks) | GPU: CUDA blocks, CPU: work groups |
| `numThreadsPerBlock` | Dimensions | Block size (threads per block) | GPU: threads per block, CPU: parallelism level |

**3. Acc Structure Components:**
- **Executor**: Defines the execution strategy (how work is dispatched to hardware)
- **FrameSpec**: Contains ThreadSpec with thread hierarchy configuration for parallel execution
- **Template Storage**: Compile-time type information for backend-specific accelerator context

#### How Executor and DeviceSpec Are Connected

Alpaka uses a **trait-based system** to determine which executors are compatible with which devices. This connection is established through compile-time compatibility checks:

**1. Compatibility Matrix (Trait System):**

| **DeviceSpec** | **API + DeviceKind** | **Compatible Executors** | **Trait Location** |
|----------------|----------------------|---------------------------|-------------------|
| `DeviceSpec{api::host, deviceKind::cpu}` | Host CPU | `exec::cpuSerial`, `exec::ompBlocks` | `api/host/Device.hpp` |
| `DeviceSpec{api::cuda, deviceKind::nvidiaGpu}` | CUDA NVIDIA GPU | `exec::gpuCuda` | `api/cuda/Device.hpp` |
| `DeviceSpec{api::hip, deviceKind::amdGpu}` | HIP AMD GPU | `exec::gpuHip` | `api/hip/Device.hpp` |
| `DeviceSpec{api::oneApi, deviceKind::intelGpu}` | OneAPI Intel GPU | `exec::oneApi` | `api/oneApi/Device.hpp` |
| `DeviceSpec{api::oneApi, deviceKind::cpu}` | OneAPI CPU | `exec::oneApi` | `api/oneApi/Device.hpp` |

**2. Trait Implementation Examples:**
```cpp
// In api/host/Device.hpp - CPU executors supported by host devices
namespace alpaka::onHost::trait {
    template<typename T_Platform>
    struct IsExecutorSupportedBy::Op<exec::CpuSerial, cpu::Device<T_Platform>> : std::true_type {};
    
    template<typename T_Platform>  
    struct IsExecutorSupportedBy::Op<exec::CpuOmpBlocks, cpu::Device<T_Platform>> : std::true_type {};
}

// In api/cuda/Device.hpp - CUDA executor supported by CUDA devices
namespace alpaka::onHost::trait {
    template<typename T_Platform>
    struct IsExecutorSupportedBy::Op<exec::GpuCuda, unifiedCudaHip::Device<T_Platform>> : std::true_type {};
}
```

**3. Runtime Compatibility Resolution:**
```cpp
// From interface.hpp - how Alpaka determines valid executor-device combinations:
constexpr auto getExecutorsList(auto const deviceSpec, auto const listOfExecutors) {
    using DeviceType = decltype(makeDeviceSelector(deviceSpec).makeDevice(0));
    using ExecutorListType = decltype(supportedExecutors(std::declval<DeviceType>(), listOfExecutors));
    return ExecutorListType{};
}

constexpr auto supportedExecutors(auto deviceHandle, auto const listOfExecutors) {
    return meta::filter([&](auto executor) constexpr {
        return trait::IsExecutorSupportedBy::Op<ALPAKA_TYPEOF(executor), ALPAKA_TYPEOF(deviceHandle)>::value;
    }, listOfExecutors);
}
```

**4. Backend Dictionary Creation Process:**
```cpp
// Step 1: For each DeviceSpec, find compatible executors
auto compatibleExecutors = getExecutorsList(deviceSpec, allExecutors);

// Step 2: Create backend dictionaries for each valid combination
auto backends = createBackendsFor(deviceSpec, compatibleExecutors);
// Results in: Dict{DictEntry{object::deviceSpec, deviceSpec}, DictEntry{object::exec, executor}}

// Step 3: Alpaka knows which executor belongs to which deviceSpec because:
//         - They are paired in the SAME backend dictionary
//         - The pairing is validated by trait::IsExecutorSupportedBy at compile time
```

**5. Why This Design Works:**
- **Compile-time Safety**: Invalid executor-device combinations are caught at compilation
- **Automatic Discovery**: Alpaka automatically finds all valid combinations without manual specification
- **Extensibility**: Adding new executors or devices only requires implementing the trait
- **Backend Isolation**: Each backend dictionary contains only compatible executor-deviceSpec pairs

**Example Usage Pattern:**
```cpp
// User code - Alpaka handles the connection automatically:
auto cfg = backend.makeDict();  // Backend dictionary with validated pairing
auto deviceSpec = cfg[object::deviceSpec];  // DeviceSpec from the dictionary
auto exec = cfg[object::exec];              // Compatible executor from SAME dictionary

// The connection is guaranteed: exec is ALWAYS compatible with deviceSpec
// because they were paired by the trait system during backend creation
```

### 3. Host Operations (`onHost/`)

#### Device and Resource Management
- **`Device.hpp`**: Device abstraction and management
  - Template-based device selection by API and device kind
  - Device handle management and lifetime
- **`DeviceProperties.hpp`**: Device capability querying
- **`DeviceSelector.hpp`**: Automatic device selection strategies

#### Execution Management
- **`Queue.hpp`**: Asynchronous task queue management
  - Backend-agnostic queue abstraction
  - Event-based synchronization
- **`Event.hpp`**: Synchronization primitives
- **`ThreadSpec.hpp`**: Thread configuration specifications

#### Memory and Data Management
- **`mem/`**: Host-side memory management
  - Buffer allocation and deallocation
  - Memory transfer operations
- **`algo/`**: Host-side parallel algorithms
  - `reduce.hpp`, `transform.hpp`: Parallel reductions and transformations
  - `concurrent.hpp`: Concurrent algorithm implementations

### 4. Accelerator Operations (`onAcc/`)

#### Accelerator Context and Execution
- **`Acc.hpp`**: Main accelerator interface and context
  - **Template-based abstraction**: `Acc<T_Storage>` wraps backend-specific storage
  - **Index and extent methods**: `getIdxWithin()`, `getExtentsOf()` for hierarchical navigation
  - **Non-copyable design**: Ensures single ownership and proper resource management
  - **Storage delegation**: Inherits from `T_Storage` for backend-specific functionality

#### Accelerator Creation Process (`makeAcc`)
The `makeAcc` function creates accelerator contexts for kernel execution:

```cpp
// Backend-specific makeAcc implementations
inline auto makeAcc(exec::CpuSerial, auto const& threadBlocking);
inline auto makeAcc(exec::CpuOmpBlocks, auto const& threadBlocking);
// Similar for GPU backends...
```

**Key characteristics**:
- **Compile-time dispatch**: Different `makeAcc` overloads for each executor
- **Thread blocking**: Takes thread specification for hierarchical parallelism
- **Backend-specific storage**: Returns `Acc<BackendSpecificStorage>`
- **Kernel execution context**: Provides the `acc` parameter passed to kernels

#### WorkGroup and Block Operations
- **`WorkGroup.hpp`**: Block-level synchronization and operations
  - Block-level barriers and atomic operations
  - Shared memory management within blocks
- **Shared memory allocation**: Static and dynamic shared memory support
- **Block-local synchronization**: `__syncthreads()` equivalent abstractions

#### Memory Operations
- **`GlobalMem.hpp`**: Global memory access patterns
- **`layout.hpp`**: Memory layout abstractions
- **`memoryFence.hpp`**: Memory synchronization primitives
- **`memoryScope.hpp`**: Memory scope definitions

#### Atomic and SIMD Operations
- **`atomic.hpp`**: Cross-platform atomic operations
  - Hierarchical atomic operations (block, device, system level)
  - Memory ordering semantics
- **`SimdAlgo.hpp`**: SIMD algorithm implementations
- **`atomicHierarchy.hpp`**: Multi-level atomic operation support

#### Kernel Execution Model
1. **Kernel definition**: User defines kernel as functor with `ALPAKA_FN_ACC`
2. **Accelerator parameter**: First parameter must be `acc` (const reference)
3. **Index navigation**: Use `acc.getIdxWithin()` and `acc.getExtentsOf()`
4. **Memory access**: Through accelerator context for proper backend mapping

### 5. Memory Management (`mem/`)

#### Memory Abstractions
- **`View.hpp`**: Multi-dimensional data view abstraction
- **`MdSpan.hpp`**: Multi-dimensional span interface
- **`DataPitches.hpp`**: Memory stride and pitch calculations

#### Iterators and Access Patterns
- **`Iter.hpp`**: Iterator abstractions for parallel access
- **`BoundaryIter.hpp`**: Boundary-aware iterators
- **`MdForwardIter.hpp`**: Multi-dimensional forward iterators

#### Index Management
- **`IdxRange.hpp`**: Index range utilities
- **`ThreadSpace.hpp`**: Thread index space management
- **`FlatIdxContainer.hpp`**: Flattened index containers

### 6. Mathematical Operations (`math/`)

#### Core Math Functions
- Platform-agnostic mathematical function abstractions
- Backend-specific optimizations for different APIs
- Complex number support through `internal/Complex.hpp`

### 7. Vector and Tensor Operations

#### Vector Types
- **`Vec.hpp`**: Multi-dimensional vector types
- **`CVec.hpp`**: Compile-time vector utilities
- **`Simd.hpp`**: SIMD vector abstractions

#### Tensor Operations
- **`tensor/`**: High-level tensor operation abstractions

## Compile-Time vs Runtime Selection Model

### Overview
Alpaka employs a sophisticated selection model that separates compile-time type decisions from runtime hardware selection, enabling both performance and flexibility.

### Compile-Time Decisions (Template Instantiation)

#### 1. Kernel Function Templates
```cpp
class VectorAddKernel {
    ALPAKA_FN_ACC auto operator()(
        auto const& acc,  // Template parameter - ACC TYPE KNOWN AT COMPILE TIME
        alpaka::concepts::MdSpan auto const A,
        alpaka::concepts::MdSpan auto const B,
        alpaka::concepts::MdSpan auto C,
        auto const& numElements) const -> void
    // Kernel is templated - instantiated for each backend
};
```

**Compile-time determinations**:
- **Kernel instantiation**: Separate template instantiation per backend
- **Memory access patterns**: Backend-specific optimizations
- **Synchronization primitives**: Backend-appropriate implementations
- **SIMD operations**: Target-specific vectorization

#### 2. Backend Dictionary Structure

The backend dictionary is a compile-time key-value structure that defines backend characteristics:

```cpp
// Each backend is a compile-time dictionary:
Dict{
    DictEntry{object::deviceSpec, DeviceSpec{api::cuda, deviceKind::nvidiaGpu}},
    DictEntry{object::exec, exec::gpuCuda}
}
```

**Dictionary Key-Value Examples:**

| **Backend Type** | **Key: object::deviceSpec** | **Key: object::exec** | **Usage Context** |
|------------------|-----------------------------|-----------------------|-------------------|
| **CUDA Backend** | `DeviceSpec{api::cuda, deviceKind::nvidiaGpu}` | `exec::gpuCuda` | NVIDIA GPU execution |
| **HIP Backend** | `DeviceSpec{api::hip, deviceKind::amdGpu}` | `exec::gpuHip` | AMD GPU execution |
| **CPU Serial** | `DeviceSpec{api::host, deviceKind::cpu}` | `exec::cpuSerial` | Single-threaded CPU |
| **CPU OpenMP** | `DeviceSpec{api::host, deviceKind::cpu}` | `exec::ompBlocks` | Multi-threaded CPU |
| **Intel OneAPI** | `DeviceSpec{api::oneApi, deviceKind::intelGpu}` | `exec::oneApi` | Intel GPU/CPU execution |

**How the Dictionary is Used:**

```cpp
// 1. Backend dictionary access (from vectorAdd.cpp line 230):
backend[alpaka::object::deviceSpec]  // Returns: DeviceSpec{api, deviceKind}
backend[alpaka::object::exec]        // Returns: Specific executor (e.g., exec::gpuCuda)

// 2. Runtime device selection using deviceSpec:
auto deviceSpec = backend[alpaka::object::deviceSpec];  // Get compile-time device spec
auto devSelector = onHost::makeDeviceSelector(deviceSpec);  // Runtime device selection
onHost::Device devAcc = devSelector.makeDevice(0);      // Select actual hardware device

// 3. Accelerator context creation using executor:
auto executor = backend[alpaka::object::exec];          // Get compile-time executor
auto acc = makeAcc(executor, threadBlocking);           // Create accelerator context
```

**Key-Value Relationship:**
- **Keys**: Predefined object types (`object::deviceSpec`, `object::exec`)
- **Values**: Backend-specific implementations (DeviceSpec combinations, execution strategies)
- **Purpose**: Compile-time backend configuration that guides runtime hardware selection and execution

### Runtime Decisions (Hardware Selection)

#### 1. Device Enumeration and Selection
```cpp
// Runtime sequence in vectorAdd example:
auto devSelector = onHost::makeDeviceSelector(deviceSpec);  // Runtime device spec
onHost::Device devAcc = devSelector.makeDevice(0);         // Runtime device selection
onHost::Queue queue = devAcc.makeQueue();                  // Runtime queue creation
```

#### 2. Backend Components Analysis (Line 230)
```cpp
backend[alpaka::object::deviceSpec], backend[alpaka::object::exec]
//  ↑                                  ↑
//  Runtime DeviceSpec                Runtime Executor
//  (API + DeviceKind)               (Execution Strategy)
```

**Backend dictionary contains**:
- **`backend[object::deviceSpec]`**: Runtime device specification
  - `DeviceSpec{api::cuda, deviceKind::nvidiaGpu}`
  - `DeviceSpec{api::host, deviceKind::cpu}`
  - `DeviceSpec{api::hip, deviceKind::amdGpu}`
- **`backend[object::exec]`**: Runtime execution strategy
  - `exec::gpuCuda`, `exec::cpuSerial`, `exec::ompBlocks`

### Selection Order and Process

#### 1. Compile-Time Phase
```cpp
// Template instantiation occurs for ALL possible backends
onHost::executeForEachIfHasDevice(
    [=](auto const& backend) { /* Template instantiated per backend */ },
    onHost::allBackends(onHost::enabledApis, onHost::example::enabledExecutors)
);
```

#### 2. Runtime Execution Sequence
```
Step 1: Backend Enumeration
   └── onHost::allBackends() creates tuples of all valid combinations
   
Step 2: Device Detection  
   └── For each backend: Check if hardware is available
   
Step 3: Device Selection
   └── makeDeviceSelector(deviceSpec) → finds matching hardware
   
Step 4: Resource Creation
   └── device.makeQueue() → creates execution context
   
Step 5: Accelerator Creation
   └── makeAcc(executor, threadBlocking) → creates kernel context
   
Step 6: Kernel Execution
   └── queue.enqueue(exec, frameSpec, kernelBundle)
```

### Backend Creation Process

#### Complete Backend Generation
```cpp
// From onHost::interface.hpp - creates all valid combinations:
constexpr auto createBackendsFor(auto const deviceSpec, auto const listOfExecutors) {
    return std::apply([deviceSpec](auto... executor) constexpr {
        return std::make_tuple(
            Dict{DictEntry{object::deviceSpec, deviceSpec}, 
                 DictEntry{object::exec, executor}}...);
    }, listOfExecutors);
}
```

#### Example Backend Tuples
```cpp
// Generated at compile time:
std::tuple<
    Dict<DeviceSpec{api::host, deviceKind::cpu}, exec::cpuSerial>,
    Dict<DeviceSpec{api::host, deviceKind::cpu}, exec::ompBlocks>,
    Dict<DeviceSpec{api::cuda, deviceKind::nvidiaGpu}, exec::gpuCuda>,
    Dict<DeviceSpec{api::hip, deviceKind::amdGpu}, exec::gpuHip>,
    // ... all combinations
>
```

### Key Design Benefits

#### 1. Performance
- **Zero runtime overhead**: Backend selection via template instantiation
- **Compile-time optimization**: Each backend gets specialized code path
- **No virtual function calls**: Direct template dispatch

#### 2. Flexibility  
- **Runtime hardware discovery**: Automatically detects available devices
- **Dynamic backend selection**: Can choose optimal backend at runtime
- **Multiple device support**: Can use different devices in same application

#### 3. Maintainability
- **Separation of concerns**: Compile-time types vs runtime hardware
- **Extensibility**: New backends add to dictionary without code changes
- **Type safety**: Compile-time verification of backend compatibility

### Accelerator Context Creation

#### makeAcc Function Role
```cpp
// In queue execution (simplified):
onAcc::Acc acc = makeAcc(executor, threadBlocking);
kernelBundle(acc);  // Pass accelerator context to kernel
```

**Process flow**:
1. **Runtime executor selection**: Based on backend dictionary
2. **Compile-time dispatch**: `makeAcc` overload resolution 
3. **Backend-specific context**: Returns `Acc<BackendStorage>`
4. **Kernel execution**: Accelerator passed to user kernel function

## Key Design Patterns

### 1. Tag-Based Dispatch
```cpp
ALPAKA_TAG(myTag);
// Used for compile-time type-safe dispatch
```

### 2. Trait-Based Customization
```cpp
namespace trait {
    template<typename T_Acc>
    struct GetName<T_Acc> {
        static auto getName() -> std::string;
    };
}
```

### 3. Handle-Based Resource Management
- RAII wrappers for backend-specific resources
- Automatic lifetime management
- Type-safe resource access

### 4. Concept-Based Interface Design
```cpp
template<alpaka::concepts::Api T_Api>
constexpr auto makeDevice(T_Api api);
```

## Component Relationships

### Dependency Hierarchy
1. **Core** → Foundation for all other components
2. **API** → Uses Core, provides backend abstraction
3. **onHost** → Uses API and Core for host-side operations
4. **onAcc** → Uses API and Core for accelerator-side operations
5. **Memory** → Uses all layers for cross-platform memory management

### Execution Flow
1. **Device Selection** → API detection and device enumeration
2. **Queue Creation** → Asynchronous execution context setup
3. **Memory Allocation** → Host and device buffer management
4. **Kernel Launch** → Accelerator execution with hierarchical parallelism
5. **Synchronization** → Event-based coordination and result retrieval

## Integration Points

### Backend Integration
Each backend provides implementations for:
- Device enumeration and management
- Queue/stream management
- Memory allocation and transfer
- Kernel compilation and execution
- Synchronization primitives

### User-Facing API
The unified interface allows users to:
- Write backend-agnostic kernel code
- Manage resources across different devices
- Express hierarchical parallelism patterns
- Optimize for different memory hierarchies