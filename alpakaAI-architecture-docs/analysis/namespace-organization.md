# Namespace organization in Alpaka Tensor Library

This page summarizes the namespaces used in the Alpaka Tensor code within this repository, where to find them, and how to use them together safely.

## Top-level

- alpaka
	- Core Alpaka execution and utilities (devices, queues, vectors, concepts, meta helpers).
	- Examples: alpaka::Vec, alpaka::exec::GpuCuda, alpaka::onHost::allocHost.

## Execution and platform namespaces

- alpaka::exec
	- Execution backends/tags (e.g., GpuCuda).
	- File: include/alpaka/executor.hpp.

- alpaka::onHost
	- Host-side algorithms, memory, and helpers used by tensor allocation and descriptors.
	- Common items: FrameSpec, ThreadSpec, allocHost, executeForEach.
	- Representative files: include/alpaka/onHost/*.

- alpaka::onAcc
	- Accelerator-side helpers (layout, atomic ops, traversal) used by device kernels.
	- Representative files: include/alpaka/onAcc/*.

## Tensor core

- alpaka::tensor
	- Core tensor types and primitives.
	- Key files:
		- TensorCore.hpp: Tensor<T, Rank, Device>, CoherenceState
		- TensorView.hpp: non-owning views
		- TensorDescriptor.hpp: metadata and layout checks
		- OpStatus.hpp: status codes for provider/generic ops interop
		- CleanTensorOpContext.hpp: runtime provider coordination

- alpaka::tensor::helpers
	- Utility helpers for tensors (tensor/Helpers.hpp).

- alpaka::tensor::generic
	- Generic tensor functionality backing the higher-level APIs (tensor/TensorGeneric.hpp).

## Operations (ops)

- alpaka::tensor::ops
	- Operation front-ends and generic implementations: Conv2D.hpp, Gemm.hpp, Activations.hpp, InferenceOps.hpp, Layer.hpp, etc.

- alpaka::tensor::ops::kernels
	- Low-level kernels used by generic ops (tensor/ops/kernels/* like GemmKernels.hpp, Conv2DKernels.hpp).

- alpaka::tensor::ops::layers
	- Layer-style compositions (Conv2DLayer, ReLULayer, LinearLayers, etc.) in tensor/ops/layers/*.

- alpaka::tensor::ops::train
	- Training-oriented ops (optimizers, backprop pieces) in tensor/ops/TrainingOps.hpp.

- alpaka::tensor::ops::clean
	- Clean/simple GEMM and related minimal APIs in tensor/ops/CleanGemm.hpp.

- alpaka::tensor::highlevel
	- Higher-level convenience APIs built on ops (tensor/ops/HighLevel.hpp).

- alpaka::tensor::ops::detail
	- Small diagnostics utilities for synchronization documentation (tensor/SyncDebug.hpp).

## Provider system

- alpaka::tensor  (providers live under this same namespace)
	- Unified provider interface and concrete vendor backends.
	- Files: tensor/providers/*
		- ProviderInterface.hpp: IOpProvider, OpType, typed/status shims and concepts
		- ProviderRegistry.hpp, BuildCaps.hpp
		- Concrete providers: CuBLASProvider.hpp, CuDNNProvider.hpp, RocBLASProvider.hpp, MIOpenProvider.hpp, DefaultProvider.hpp

## Documentation helpers

- alpaka::tensor::docs
	- Queue semantics and documentation-only helpers (tensor/QueueSemantics.hpp).

## Quick reference: using patterns

- Prefer qualified names in headers to avoid ADL surprises:
	- alpaka::tensor::Tensor2D<float, Device>
	- alpaka::tensor::ops::gemm(exec, device, queue, ...)

- In source files, use targeted using only at small scope:
	- using alpaka::tensor::ops::gemm;
	- Avoid using namespace alpaka; at file scope in library code.

- Providers and context:
	- alpaka::tensor::IOpProvider and CleanTensorOpContext are in alpaka::tensor.

## Minimal examples

- Create and move a tensor to device:

	```cpp
	using Exec = alpaka::exec::GpuCuda; // or other backend
	using Device = /* your device type */;
	using Queue = /* your queue type */;

	Device device = /* ... */;
	Queue queue   = /* ... */;

	alpaka::tensor::Tensor2D<float, Device> x(device, {64, 64}, "x");
	x.ensureOnDevice(device, queue);
	```

- Run GEMM via the ops front-end:

	```cpp
	using namespace alpaka::tensor;
	using namespace alpaka::tensor::ops;

	Tensor1D<float, Device> A(device, {M*K});
	Tensor1D<float, Device> B(device, {K*N});
	Tensor1D<float, Device> C(device, {M*N});

	gemm(exec, device, queue, 'N', 'N', M, N, K, 1.0f, A, B, 0.0f, C);
	```

- Provider interface at runtime:

	```cpp
	alpaka::tensor::CleanTensorOpContext<Exec, Device, Queue> ctx(exec, device, queue);
	auto& prov = ctx.getGemmProvider();
	if (prov.isActive() && prov.supportsOperation(alpaka::tensor::OpType::GEMM)) {
			// try vendor path via status API
			prov.gemm_status(exec, device, queue, M, N, K, 1.0f, A, B, 0.0f, C);
	}
	```

## Notes

- Most tensor headers live under include/alpaka/tensor/... and consistently use namespace alpaka::tensor with nested namespaces for ops (::ops), kernels (::ops::kernels), and layers (::ops::layers).
- Execution helper namespaces alpaka::onHost and alpaka::onAcc are part of the broader Alpaka project and are used by tensor code for allocation, iteration, and kernel support.

