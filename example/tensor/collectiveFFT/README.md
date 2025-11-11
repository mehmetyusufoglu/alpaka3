# Tensor Collective Demo

This example demonstrates Alpaka's experimental NCCL collective provider.
# Tensor Collective FFT

`tensorCollectiveFFT` demonstrates how to assemble the discrete Fourier transform
of a very large 1-D signal by orchestrating Alpaka tensors, NCCL collectives, and
optional MPI bootstrapping. Each rank/GPU owns every `worldSize`-th sample of the
global signal, computes a naive local DFT, applies the offset-dependent phase, and
contributes its slice to an `ncclAllReduce`. Rank 0 validates the reconstructed
spectrum against a direct O(n²) DFT (or a file-backed reference) so we can confirm
the multi-device composition is numerically correct.

## Build Requirements

- CUDA backend enabled (`alpaka_ENABLE_CUDA=ON`) and at least one CUDA GPU per
  participating rank.
- Collective provider enabled with NCCL support (`alpaka_ENABLE_COLLECTIVES=ON`,
  `alpaka_ENABLE_NCCL=ON`).
- MPI is optional but strongly recommended for multi-process NCCL bootstrap on
  clusters (`alpaka_ENABLE_MPI=ON`).

Configure and build:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
      -Dalpaka_ENABLE_COLLECTIVES=ON -Dalpaka_ENABLE_NCCL=ON
cmake --build build --target tensorCollectiveFFT -j8
```

Make sure `NCCL_ROOT` is visible to CMake (environment variable or
`CMAKE_PREFIX_PATH`) and `LD_LIBRARY_PATH` contains the NCCL runtime when running
the executable.

## Command-Line Options

```
--signal-length=<N>     Total samples in the global 1-D signal (default 16384).
--signal-file=<path>    Binary float32 file with N samples for the real-valued signal.
--preview-bins=<K>      Number of leading frequency bins to print (default 12).
--skip-verify           Disable direct DFT verification.
--verify-direct         Force direct O(N²) verification (default: enabled).
--verify-direct-precision=float|double  Precision for the naive DFT (default double).
--reference-fft=<path>  Binary float32 pairs (real, imag) reference spectrum for comparison.
--verify-abs=<eps>      Absolute tolerance for verification (default 1e-4).
--verify-rel=<eps>      Relative tolerance for verification (default 1e-3).
--disable-provider-fft  Skip the cuFFT-backed provider and use the naive host DFT.
--force-provider-fft    Require the cuFFT provider; exit if cuFFT is unavailable.
```

By default the sample automatically enables the cuFFT-backed tensor provider when
CUDA and cuFFT are both available at runtime. If either dependency is missing the
context transparently falls back to the naive host DFT unless `--force-provider-fft`
is specified.

When `--signal-file` is omitted, each rank synthesizes a deterministic multi-tone
signal so the demo can run without external data.

## Running Locally (single process)

Once built, launch the executable from the build tree:

```bash
./example/tensor/collectiveFFT/tensorCollectiveFFT --signal-length=32768
```

If only one CUDA device is visible the program still forms a collective group of
size 1; NCCL all-reduce becomes a no-op but the verification still validates the
pipeline. Use `--skip-verify` while experimenting with very long signals to avoid
the naive O(n²) reference cost.

## Multi-Rank Execution

When MPI is present the sample uses it to bootstrap NCCL across processes. Typical
launch sequence on a Slurm cluster (two nodes, one rank per GPU):

```bash
module load gcc/12.2.0 cuda/12.4 nvidia/24.3 openmpi/4.1.5-cuda12x-gdr
export NCCL_ROOT=$NVHPC/Linux_x86_64/24.3/comm_libs/nccl
export LD_LIBRARY_PATH=$NCCL_ROOT/lib:$LD_LIBRARY_PATH

salloc -N2 -n4 --partition=casus_a100 --gres=gpu:2 --time=00:30:00
srun --jobid=$SLURM_JOB_ID --pty bash -l
module load cuda/12.4
scontrol show hostnames "$SLURM_JOB_NODELIST" | awk '{print $0 " slots=2"}' > hostfile
exit

mpirun -n 4 --hostfile hostfile --map-by ppr:2:node --bind-to none \
       -x NCCL_ROOT -x LD_LIBRARY_PATH \
       ./example/tensor/collectiveFFT/tensorCollectiveFFT --signal-length=65536
```

Rank 0 prints the first few bins of the assembled spectrum, the verification
results, and the final success banner. All ranks log which strided sample indices
they own so it is easy to confirm the partitioning.

## Verification Workflow

We reconstruct the full FFT using the formula

```
X[k] = Σ_{r=0}^{P-1} exp(-j·2π·r·k/N) · S_r[k mod M]
```

where `P` is the participating rank count and `M = N / P`. After the NCCL
all-reduce every process holds the same global spectrum `X`. Rank 0 then:

1. Rebuilds the contiguous signal (from file or synthetic generator).
2. Runs a naive O(N²) DFT to produce a direct reference spectrum.
3. Optionally loads a binary reference FFT supplied via `--reference-fft`.
4. Compares each reference against `X` with tolerances `--verify-abs` and
   `--verify-rel`, printing per-source pass/fail summaries.

Adjust tolerances as needed to match the precision of upstream generators. Disable
the direct check with `--skip-verify` when using very large signals or when an
external FFT provides the baseline.

## Troubleshooting Notes

- Missing CUDA/NCCL: the executable falls back to host executors and prints a
  message explaining why the collective path was skipped.
- NCCL bootstrap errors: check that all ranks agree on `worldSize`, `worldRank`,
  and receive the same `ncclUniqueId`. MPI warnings usually indicate the launch
  command is not running inside a valid allocation.
- Verification failures: inspect the reported bin index, ensure the reference file
  matches the chosen signal length, and verify the tolerances are appropriate for
  the numeric precision in use.

### Build Architecture Mismatch

If you compile on a login node whose CPU architecture differs from the compute
nodes, the resulting binary can fail immediately with `Exec format error` when
`mpirun` starts the ranks. Rebuild while you are inside the Slurm allocation
(`srun --pty …`) using the same module stack that will be active during execution.
You can confirm the artifact targets x86-64 by running `file
build/example/tensor/collectiveFFT/tensorCollectiveFFT`; on Hemera it should
report `ELF 64-bit LSB executable, x86-64`. If you must build on the login node,
ensure the toolchain emits binaries for the compute-node architecture explicitly.

Local size: 2
Collective provider for NCCL is only available on CUDA executors; skipping.

=== Tensor Collective Demo ===
API: Host
Executor: alpaka::exec::CpuSerial
World size: 4
Local size: 2
Collective provider for NCCL is only available on CUDA executors; skipping.

# Example output on GPU-enabled nodes (NCCL active)

=== Tensor Collective Demo ===
API: Cuda
Executor: alpaka::exec::GpuCuda
World size: 4
Local size: 2
Rank 0 using device 0
Rank 1 using device 1
Rank 3 using device 1
Rank 2 using device 0
Rank 0 result: 4 8 12 16
Rank 1 result: 4 8 12 16
Rank 2 result: 4 8 12 16
Rank 3 result: 4 8 12 16


Some diagnostics commands:

echo $SLURM_JOB_ID → empty means you’re not in an srun --pty shell.
squeue -u yusufo81 → if the allocation is gone, there’s no job listed.
hostname → compute nodes show ga0XX; login node is hemera5.cluster.
So seeing [yusufo81@hemera5 build]$ doesn’t imply you’re still inside srun; it’s the normal login-shell prompt.

A CORRECT HISTORY

[yusufo81@hemera5 build]$  srun --jobid=$SLURM_JOB_ID --pty bash -l
[yusufo81@ga010 build]$ module load cuda/12.4
Unloading module cuda/12.4 
Loading module cuda/12.4 
[yusufo81@ga010 build]$     scontrol show hostnames "$SLURM_JOB_NODELIST"
ga010
ga012
[yusufo81@ga010 build]$  mpirun -n 4 \
>          --hostfile hostfile \
>          --map-by ppr:2:node --bind-to none \
>          --oversubscribe \
>          -x NCCL_ROOT -x LD_LIBRARY_PATH \
>          ./build/example/tensor/collectiveDemo/tensorCollectiveDemo
--------------------------------------------------------------------------
A hostfile was provided that contains at least one node not
present in the allocation:

  hostfile:  hostfile
  node:      ga011

If you are operating in a resource-managed environment, then only
nodes that are in the allocation can be used in the hostfile. You
may find relative node syntax to be a useful alternative to
specifying absolute node names see the orte_hosts man page for
further information.
--------------------------------------------------------------------------
--------------------------------------------------------------------------
An internal error has occurred in ORTE:

[[41244,0],0] FORCE-TERMINATE AT (null):1 - error plm_slurm_module.c(475)

This is something that should be reported to the developers.
--------------------------------------------------------------------------
[yusufo81@ga010 build]$ ls
alpaka_build_files  alpakaFeatureTests  CMakeFiles           CTestTestfile.cmake  hostfile
alpakaConfig.cmake  CMakeCache.txt      cmake_install.cmake  example              Makefile
[yusufo81@ga010 build]$ vi hostfile 
[yusufo81@ga010 build]$  mpirun -n 4          --hostfile hostfile          --map-by ppr:2:node --bind-to none          --oversubscribe          -x NCCL_ROOT -x LD_LIBRARY_PATH          ./build/example/tensor/collectiveDemo/tensorCollectiveDemo

ex^C
^C[yusufo81@ga010 build]$ exit
logout
srun: error: ga010: task 0: Exited with exit code 1
[yusufo81@hemera5 build]$     mpirun -n 4          --hostfile hostfile          --map-by ppr:2:node --bind-to none          --oversubscribe          -x NCCL_ROOT -x LD_LIBRARY_PATH          ./example/tensor/collectiveDemo/tensorCollectiveDemo

=== Tensor Collective Demo ===
API: Host
Executor: alpaka::exec::CpuOmpBlocks
World size: 4
Local size: 2
Collective provider for NCCL is only available on CUDA executors; skipping.

=== Tensor Collective Demo ===
API: Host
Executor: alpaka::exec::CpuSerial
World size: 4
Local size: 2
Collective provider for NCCL is only available on CUDA executors; skipping.

=== Tensor Collective Demo ===
API: Cuda
Executor: alpaka::exec::GpuCuda
World size: 4
Local size: 2
Rank 0 using device 0
Rank 1 using device 1
Rank 2 using device 0
Rank 3 using device 1
Rank 2 result: 4 8 12 16
Rank 0 result: 4 8 12 16
Rank 1 result: 4 8 12 16
Rank 3 result: 4 8 12 16
[yusufo81@hemera5 build]$ 
