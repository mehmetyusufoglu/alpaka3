# Performance of End-to-End Pipelines (Inference and Training)

This document analyzes performance characteristics and common bottlenecks for CNN pipelines implemented with the Alpaka Tensor Library. We split analysis along two axes: scale (inter-layer vs intra-layer) and implementation path (Alpaka generic kernels vs vendor libraries).

- Pipelines: Inference and Training
- Scales:
  - Inter-layer (between layers): submission order, waits, fusion, memory reuse, synchronization
  - Intra-layer (inside a layer): kernel configuration, memory access, tiling, vendor call knobs
- Intra-layer subcases:
  - A) Alpaka kernels
  - B) Vendor libraries (cuBLAS/cuDNN/MIOpen/rocBLAS)

References in code: `QueueSemantics.hpp`, `ops/{Conv2D,Gemm,Activations,InferenceOps}.hpp`, `ops/kernels/*`, `providers/*`, `CleanTensorOpContext.hpp`.

## 1) Inter-layer scale (between layers)

Typical CNN chain (inference): Conv2D → BatchNorm → ReLU → Pool → ...
Training adds: Gradients (dConv, dBN, dAct), Optimizer updates, Data transfers.

Key factors:

- Queue ordering vs explicit waits
  - Alpaka queues are async from the host, ordered on the device. Enqueueing Conv→BN→ReLU on the same queue will respect dependencies automatically.
  - Bottleneck: calling `::alpaka::onHost::wait(queue)` after each layer (or `toHost()` between layers) stalls the pipeline. Many ops already avoid immediate wait; host access synchronizes implicitly.
  - Guidance: batch multiple layer ops on the same queue; defer waits until a materialization boundary (I/O, final readback, or cross-queue dependency).

- Missing kernel merging / fusion opportunities
  - Separate Conv→BN→ReLU incur multiple kernel launches and global memory round-trips.
  - Mitigation:
    - Provide fused kernels or layer-level fusers (Conv+BN fold, Conv+Bias+ReLU epilogues).
    - Use `ops::BatchNormFold` during inference to statically fold BN into Conv weights/bias when possible.
    - Prefer provider epilogues when supported (e.g., cuDNN fused ops, GEMM epilogues in BLAS for bias/relu).

- Intermediate memory traffic and reuse
  - Allocating distinct output tensors per layer increases memory bandwidth pressure and allocator churn.
  - Mitigation:
    - Reuse buffers across layers with identical shapes; keep a small tensor pool.
    - Favor in-place variants (e.g., `relu_inplace`) when correctness allows.
    - Avoid unnecessary `toHost()`; keep tensors on device between layers.

- Synchronization with host
  - Patterns like debug validations or eager `toHost()` force host-device synchronization.
  - Flags: `ALPAKA_OPS_VERBOSE` can add waits for timing in some paths; ensure it's disabled for perf runs.
  - Mitigation: group multiple checks after a single `wait()`; use device-side diagnostics when possible.

- Multi-queue usage
  - Using different queues per layer can enable overlap, but without explicit dependencies may require waits.
  - Start with a single queue pipeline; introduce additional queues only with explicit ordering or events.

- Provider context transitions
  - Switching between vendor and generic paths across layers can change performance characteristics (different optimal tensor layouts or temporary allocations).
  - Mitigation: stick to a consistent provider path per device when possible; ensure descriptors/layouts are contiguous (NCHW) to avoid adapter overheads.

Inference-specific considerations:
- Prefer BN folding (eliminate BN layer at runtime), reduce memory traffic, reduce kernel count.
- Prefer epilogue-fused activations when provider supports them.

Training-specific considerations:
- Extra data dependencies (activations/residuals) increase memory footprint; ensure buffer reuse for saved activations.
- Overlap data loading/host preprocessing with compute (separate queue/thread if needed), but maintain correct ordering.
- Avoid host-side gradient checks for every step; batch checks.

## 2) Intra-layer scale — A) Alpaka kernels

Applies to ops implemented with generic kernels: e.g., naive/tiled Conv2D in `ops/Conv2D.hpp`, elementwise activations, pooling kernels, BatchNorm kernels.

Primary issues and mitigations:

- Kernel launch geometry and FrameSpec
  - Mismatch between `numFrames` and `frameExtent` can yield poor occupancy.
  - Ensure 4D kernels use meaningful extents (e.g., mapping over [N,C,H,W] in Conv and BN).

- Tiling and shared memory usage (Conv2D)
  - Condition: tiled path enabled only when stride/dilation=1 and kernel <= bounds, and for supported backends (e.g., CUDA by default).
  - Env override: `ALPAKA_CONV2D_FORCE_TILED=1` for testing.
  - Choose tile sizes to balance occupancy and shared memory (e.g., 16x16 with halos up to 7x7).

- Memory access patterns
  - Favor coalesced/global contiguous access; keep tensors contiguous row-major (NCHW) — see `TensorDescriptor` assertions.
  - Use iterators/index maps (`onAcc::makeIdxMap`) that traverse in memory-friendly order; consider `layout::Optimized` if available.

- Synchronization inside ops
  - Some kernels previously forced waits (for timing or safety). Prefer asynchronous enqueue; only synchronize on data hazards at boundaries.

- Temporary allocations
  - Repeated device allocations inside layer functions (e.g., output tmp for bias add) add overhead.
  - Pre-allocate and reuse temporaries or use in-place algorithms where safe.

- Precision and math intrinsics
  - Activate fast math if acceptable; avoid unnecessary double conversions in tight loops.
  - For softmax, numerically stable algorithm may require multiple passes; consider warp-level reductions when available.

- Diagnostics overhead
  - `ALPAKA_OPS_VERBOSE` or debug descriptor checks can significantly slow kernels; keep disabled for perf.

Training-specific intra-layer notes:
- Backward kernels require additional reads/writes; ensure symmetric tiling and reuse intermediate results.
- Accumulation precision (e.g., FP32 vs FP16 accumulation) can affect both speed and convergence.

## 3) Intra-layer scale — B) Vendor libraries

Covers cuBLAS/cuDNN/rocBLAS/MIOpen via providers.

Key performance factors:

- Handle/stream binding
  - Ensure cuBLAS/rocBLAS handle is set to the current queue’s native stream for every call (code paths do this; verify in multi-queue scenarios).

- Algorithm selection and math mode
  - cuBLAS: mixed precision and tensor cores — control with `ALPAKA_USE_FP16` or TF32 modes; disable via `ALPAKA_DISABLE_CUBLAS`.
  - cuDNN/MIOpen: algorithm selection often has “fastest” vs “deterministic/safe” flags; choosing safer algorithms may reduce speed.
  - Expose a knob in provider to pick autotuned/heuristic vs deterministic modes if not present.

- Epilogues and fusion
  - Prefer fused epilogues (bias, activation) in GEMM/Conv when provider supports them to reduce memory traffic and kernel launches.

- Workspace and temporary memory
  - Conv/BN often require workspace; ensure adequate workspace allocation and reuse to avoid repeated malloc/free.
  - Provide a memory pool for provider workspaces if possible.

- Layout and descriptors
  - Mismatched layout (e.g., NCHW vs NHWC) can force internal transforms; keep tensors contiguous NCHW unless provider prefers otherwise.

- Batch size and problem size sensitivity
  - Vendor libraries may have performance cliffs for small sizes (e.g., small N, small HxW). Consider batched kernels or custom fast paths for micro sizes.

- Error handling paths
  - Falling back to safer algorithms or status-based paths after an exception can hide performance regressions. Log once and allow configuration to force a path.

Training-specific vendor notes:
- Deterministic vs nondeterministic kernels (reproducibility) — deterministic is often slower; make policy explicit.
- Mixed precision training: balance speedups with loss scaling and accumulation precision.

## 4) Concrete checklists and mitigations

Inter-layer (both inference/training):
- [ ] Avoid `wait()` and `toHost()` between layers unless necessary
- [ ] Keep a single queue for the pipeline unless explicitly overlapping
- [ ] Reuse buffers; prefer in-place activations and fused operations
- [ ] Fold BatchNorm into Conv for inference
- [ ] Disable verbose/debug checks in performance runs

Intra-layer (Alpaka kernels):
- [ ] Choose good FrameSpec and tiling; enable tiled conv when applicable
- [ ] Ensure contiguous NCHW; coalesced access, optimized iteration
- [ ] Minimize temp allocations and synchronize lazily

Intra-layer (Vendor libraries):
- [ ] Bind handles to queue stream; ensure correct math mode
- [ ] Select fast algorithms; allow deterministic/safe override
- [ ] Reuse workspaces; enable fused epilogues
- [ ] Align layouts with provider preferences

## 5) Pipeline-specific notes

Inference:
- Hot path optimization: fold BN, fuse activations, minimize kernel count
- Benchmark critical batch sizes; consider dynamic tiling thresholds

Training:
- Overlap data pipeline with compute; minimize host sync
- Optimize backward ops similarly; reuse forward activations efficiently
- Make deterministic mode a toggle with clear perf implications

## 6) What to measure

- End-to-end latency and throughput (images/s)
- Per-layer time breakdown; kernel occupancy and achieved bandwidth
- Synchronization points (number and duration of waits)
- Memory allocation counts and bytes; workspace reuse efficiency
- Provider vs generic path ratios; autotuned algorithm choices

## 7) Next steps

- Add optional “lazy sync” mode across the library to defer waits until host access
- Add Conv+Bias(+ReLU) fused variant and BN folding utility in highlevel
- Introduce a simple workspace/buffer pool shared across ops
- Expose provider knobs (deterministic vs fastest; autotune) via CleanTensorOpContext config
- Provide micro-benchmarks per op and end-to-end pipelines; document perf baselines per backend
