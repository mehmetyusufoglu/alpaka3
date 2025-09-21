# Execution Model & Providers

## Execution Triplet
- Every op takes `(exec, device, queue)`
- Queue semantics: async submission, ordered execution (see `QueueSemantics.hpp`)

## Providers
- `IOpProvider` abstracts vendor libraries
- `ProviderRegistry` chooses provider per Exec: cuBLAS/cuDNN on CUDA, rocBLAS/MIOpen on HIP, else Default
- `DefaultProvider` implements generic Alpaka kernels

## CleanTensorOpContext
- Coordinates providers and offers typed methods
- Methods: `gemm`, `conv2d`, `batchnorm`, `pooling` (through providers)
- Falls back to generic kernels when provider reports Unsupported

## Backend Capabilities
- Conv2D tiled path enabled by `Conv2DBackendCapabilities<Exec>::supportsTiledOptionB` (CUDA defaults to true)
- GEMM prefers cuBLAS when available; env flags: `ALPAKA_DISABLE_CUBLAS`, `ALPAKA_USE_FP16`

## Provider selection: compile-time vs runtime

- Compile-time selection (templates):
  - `select_gemm_provider<Exec>::type` -> `CuBLASProvider | RocBLASProvider | DefaultProvider`
  - `select_conv_provider<Exec>::type` -> `CuDNNProvider | MIOpenProvider | DefaultProvider`
  - Decides the provider class type based on `Exec` and build-time caps (`ALPAKA_HAS_CUBLAS`, `ALPAKA_HAS_CUDNN`, `ALPAKA_HAS_ROCBLAS`, `ALPAKA_HAS_MIOPEN` in `BuildCaps`).
- Runtime activation (instances):
  - `CleanTensorOpContext` constructs provider instances and checks `isActive()` and `supportsOperation(op)` per call path
  - If typed fast-path available (e.g., cuBLAS/cuDNN), calls typed API; else uses `*_status(...)` type-erased calls; if still unsupported -> falls back to generic ops
  - Environment flags influence behavior:
    - `ALPAKA_DISABLE_CUBLAS=1` disables cuBLAS use even on CUDA
    - `ALPAKA_USE_FP16=1` may enable FP16 mixed-precision GEMM (Tensor Cores) in cuBLAS path
    - `ALPAKA_CONV2D_FORCE_TILED=1` forces tiled Conv2D Option B when possible
    - `ALPAKA_OPS_VERBOSE=1` prints detailed op decisions and timings

See diagrams:
- `diagrams/provider-selection-overview.puml`
- `diagrams/provider-decision-flow.puml`
