# Practical Usage — Alpaka Tensor

Recipes derived from examples/benchmarks and unit tests.

## Setup
- Include umbrella: `#include <alpaka/alpaka.hpp>`
- Create device, queue, exec (CPU/GPU)

## Tensor Creation
- `Tensor4D<float, Device> x(device, {N,C,H,W});`
- Or via helpers: `auto x = helpers::makeTensor4D<float>(device, N,C,H,W, "x");`

## GEMM
- `ops::gemm(exec, device, queue, 'N','N', M,N,K, 1.f, A,B, 0.f, C);`
- High-level: `highlevel::gemm(exec, device, queue, M,N,K, 1.f, A,B, 0.f, C);`

## Conv2D
- `ops::conv2d(exec, device, queue, input, weight, params);`
- `auto p = highlevel::make_conv2d_params(1,1,1,1);`
- High-level: `highlevel::conv2d(exec, device, queue, input, weight, p);`

## ReLU
- Out-of-place: `ops::relu(exec, device, queue, in, out);`
- In-place: `ops::relu_inplace(exec, device, queue, t);`

## Pooling
- `ops::max_pool2d(exec, device, queue, x, {kH,kW,sH,sW,pH,pW});`
- `ops::avg_pool2d(exec, device, queue, x, {kH,kW,sH,sW,pH,pW});`

## BatchNorm (Inference)
- `ctx.batchnorm(x, mean, var, gamma, beta, eps);` using `CleanTensorOpContext`
- Or fallback kernel via `BatchNorm2DLayerStruct` in Layers

## Layers Composition
- See `tests/unit/tensor/residualConnections.cpp` for BasicBlock composition

## Tips
- Avoid unnecessary `wait()`; queue ordering ensures deps
- Call `toHost()` only when CPU needs data; it synchronizes
- Use env flags for debugging perf: `ALPAKA_OPS_VERBOSE`, `ALPAKA_DISABLE_CUBLAS`
