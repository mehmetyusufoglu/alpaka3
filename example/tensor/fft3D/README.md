# tensorFft3D

`tensorFft3D` demonstrates how to execute batched 3-D complex FFTs through the
Alpaka tensor provider interface. When CUDA and cuFFT are available the example
runs the transforms on the GPU via the new `CuFFTProvider`. Otherwise it falls
back to the generic host implementation.

## Build

```bash
cmake --build build --target tensorFft3D -j8
```

## Usage

```bash
./example/tensor/fft3D/tensorFft3D [options]
```

Options:

- `--size=<N>`: Set identical dimensions for x/y/z (default 16).
- `--nx=<N>`, `--ny=<N>`, `--nz=<N>`: Override individual extents.
- `--batch=<B>`: Number of independent transforms per launch (default 1).
- `--no-verify`: Skip the inverse-transform accuracy check.
- `--verbose`: Print the first few frequency-domain samples.
- `--force-provider-fft`: Require cuFFT; the program exits with an error if the
  provider is unavailable.
- `--real-signal` / `--signal-type=real`: Treat the input volume as real-valued
  and execute cuFFT's R2C/C2R path (default is complex data).

The example reports the maximum absolute and relative error after performing a
forward + inverse transform pair. When cuFFT is not present the fallback path
executes a naive O(N²) transform on the host, which is only practical for the
default (16³) problem size.

## Running on Hemera with `srun`

The Hemera login nodes and compute nodes differ in architecture, so build and
run the sample inside your Slurm allocation:

```bash
# Login node: load toolchain and capture NCCL for cuFFT-enabled builds.
module load gcc/12.2.0 cuda/12.4 nvidia/24.3 openmpi/4.1.5-cuda12x-gdr
export NCCL_ROOT=$NVHPC/Linux_x86_64/24.3/comm_libs/nccl
export LD_LIBRARY_PATH=$NCCL_ROOT/lib:$LD_LIBRARY_PATH

# Request a single-GPU allocation and open a shell on the compute node.
salloc -N1 -n1 --partition=casus_a100 --gres=gpu:1 --time=00:30:00
srun --jobid=$SLURM_JOB_ID --pty bash -l

# Inside the allocation: rebuild for the compute-node architecture.
module load cuda/12.4
cd ~/alpaka3
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -Dalpaka_ENABLE_COLLECTIVES=ON -Dalpaka_ENABLE_NCCL=ON
cmake --build build --target tensorFft3D -j8

# Launch the demo with srun so the GPU resource is correctly bound.
cd ~/alpaka3/build
srun -n 1 --gpus-per-task=1 ./example/tensor/fft3D/tensorFft3D --size=64 --verbose
```

Adjust `--size`/`--nx`/`--ny`/`--nz` and `--batch` to explore different problem
shapes. Use `--no-verify` for very large grids if the inverse-check cost becomes
prohibitive.

For a rectangular signal matching your 672 × 1344 × 607 grid, invoke the
real-valued path to reduce GPU memory pressure:

```bash
srun -n 1 --gpus-per-task=1 ./example/tensor/fft3D/tensorFft3D \
  --nx=672 --ny=1344 --nz=607 --signal-type=real --no-verify --verbose
```

The `--no-verify` flag avoids the costly O(N²) host check, which is impractical
for 548 million voxels. Drop the flag once you validate smaller problem sizes.
