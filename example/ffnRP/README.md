# ffnRP – Transformer Feed-Forward (FFN) Reduced Precision Benchmark

Implements the two-layer feed-forward network used in Transformer blocks (as in MLPerf BERT/GPT):

```
H = GELU(X * W1 + b1)
Y = H * W2 + b2
```

Shapes:
* `X` : [B*S, H]
* `W1`: [H, 4H]
* `W2`: [4H, H]
* Bias vectors: `b1` [4H], `b2` [H]

Estimated FLOPs per run (ignoring bias & GELU minor cost): `16 * (B*S) * H^2`.

Supports precision types: float, double, TF32, BF16, FP16 and an `auto` policy.

Example:

```
./ffnRP --hidden 1024 --batch 8 --seq 128 --type bf16 --repeats 10
```

This example is intended to highlight performance differences in compute-bound mixed / reduced precision workloads beyond memory-bound vector operations.
