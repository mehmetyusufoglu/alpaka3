# Tensor Collective Demo

This example demonstrates Alpaka's experimental NCCL collective provider.

- Configures a `CleanTensorOpContext` on the first CUDA device.
- Runs an in-place NCCL all-reduce to verify the provider wiring.
- Falls back cleanly when CUDA or NCCL is missing.
- Uses MPI (when available) to bootstrap multi-process NCCL communicators.

## Getting NCCL Running

1. **Detect NCCL at configure time**
    - Export `NCCL_ROOT`, `NCCL_PATH`, or extend `CMAKE_PREFIX_PATH` so CMake can locate the headers and library (`libnccl.so`).
    - Clear the CMake cache or reconfigure from a clean build directory.
    - Run `cmake -S .. -B build -Dalpaka_ENABLE_COLLECTIVES=ON -Dalpaka_ENABLE_NCCL=ON` and confirm the log prints the resolved NCCL include and library paths.

2. **Propagate headers and libraries to targets**
    - After configuration, rebuild `tensorCollectiveDemo` with `cmake --build build --target tensorCollectiveDemo VERBOSE=1`.
    - Check the `nvcc` command line for `-DALPAKA_HAS_NCCL` and the NCCL include directory; verify the link step lists `libnccl.so`.

3. **Ensure runtime visibility**
    - Add the NCCL library directory to `LD_LIBRARY_PATH` or rely on the rpath set by CMake so `libnccl.so` loads successfully.
    - Set environment tuning knobs (e.g., `NCCL_P2P_DISABLE=0`, `NCCL_SOCKET_IFNAME`) only if your fabric requires them.

4. **Scale beyond a single rank**
    - Build with MPI available on your system. CMake picks up `MPI::MPI_CXX`/`MPI::MPI_C` automatically and enables NCCL bootstrap support.
    - Launch the demo with one process per GPU (e.g., `srun -N2 -n4 bash -lc 'mpirun -n 4 ./tensorCollectiveDemo'`). Rank 0 generates a NCCL unique ID and broadcasts it with MPI.
    - Ensure each rank exposes exactly one CUDA device (set `CUDA_VISIBLE_DEVICES` or rely on Slurm binding) so the demo maps MPI ranks to GPUs.
    - Verify the output prints one line per rank with the reduced values. The vector entries should equal the sum of all ranks when NCCL is active.
