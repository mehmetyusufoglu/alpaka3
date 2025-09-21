# API Surface — Alpaka Tensor Library

Key headers and APIs to use the tensor subsystem.

## Aggregation Header
- `#include <alpaka/tensor.hpp>` — pulls all tensor components

## Core Types
- `tensor::Tensor<T, Rank, Device>` and aliases `Tensor{1D..4D}`
- `tensor::TensorView<T, Rank, Device>`
- `tensor::CoherenceState`

## Helpers
- `helpers::makeTensor{1D..4D}` and `helpers::fillHost`
- `helpers::Context{device, queue}`

## High-Level Ops
- `highlevel::gemm`, `highlevel::conv2d`, `highlevel::relu`, `highlevel::relu_inplace`

## Ops (Detailed)
- `ops::gemm(exec, device, queue, ...)`
- `ops::conv2d(exec, device, queue, input4d, weight4d, params)`
- `ops::relu(exec, device, queue, in, out)` and `relu_inplace`
- `ops::avg_pool2d`, `ops::max_pool2d`, `ops::softmax2d`, `ops::linear`, `ops::bias_add_*`

## Provider Context
- `CleanTensorOpContext<Exec, Device, Queue>` exposes:
  - `gemm(M,N,K,alpha,A,B,beta,C)`
  - `conv2d(input, weight, params)`
  - `batchnorm(input, mean, var, gamma, beta, eps)`

## Example Shapes
- Conv2D: Input[N,C_in,H,W], Weight[C_out,C_in,K_h,K_w], Output[N,C_out,H_out,W_out]
- GEMM: A[M*K], B[K*N], C[M*N] — row major
