# Ops Catalog — Alpaka Tensor

Overview of major operations and their contracts.

## Conv2D
- Input: `Tensor4D<T> [N,C_in,H,W]`, Weight: `Tensor4D<T> [C_out,C_in,K_h,K_w]`
- Output: `Tensor4D<T> [N,C_out,H_out,W_out]`
- Params: stride, pad, dilation
- Backends: generic, tiled CUDA path, cuDNN via provider (future)

## GEMM
- Row-major: A[M*K], B[K*N], C[M*N]
- Default provider: generic kernel; CUDA: cuBLAS (with Tensor Cores)

## Activations
- `relu(in,out)`, `relu_inplace(t)`
- SIMD and generic kernels

## Pooling
- `max_pool2d`, `avg_pool2d`, `GlobalAveragePool2DLayerStruct`
- Params: kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w

## BatchNorm (Inference)
- Inputs: runningMean[C], runningVar[C], gamma[C], beta[C]
- Fused per-element transform with eps
- Provider delegation when available (cuDNN/MIOpen) else generic kernel

## Inference Ops
- BiasAdd (2D/4D), Linear (GEMM+bias), Softmax (row-wise stable)

## Layers
- Conv2DLayerStruct, ReLU, MaxPool/AvgPool, GlobalAvgPool2D, BatchNorm2D, BasicBlock
- Compose via operator()(exec, device, queue, input)
