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

ExampleNCCL Smoke Test Steps (MPI bootstrap)

module load gcc/12.2.0 cuda/12.4 nvidia/24.3 (keeps nvcc happy and exposes NCCL)
export NCCL_ROOT=$NVHPC/Linux_x86_64/24.3/comm_libs/nccl
export CMAKE_PREFIX_PATH=$NCCL_ROOT:$CMAKE_PREFIX_PATH
export LD_LIBRARY_PATH=$NCCL_ROOT/lib:$LD_LIBRARY_PATH
rm -rf build && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target tensorCollectiveDemo -j8
Run across two nodes (1 GPU per rank)

salloc -N2 -n4 --partition=gpu_v100_low --time=00:30:00
srun -N2 -n4 bash -lc 'mpirun -n 4 ./build/example/tensor/collectiveDemo/tensorCollectiveDemo'
Each rank should print identical vectors; sum should equal worldSize * [1,2,3,4]. Use scontrol show job $SLURM_JOB_ID (before exit) to confirm two node IDs. Let me know the output so we can capture it for the report.

Another example:

# 1. Check if MPI was found at all
grep -E "MPI_FOUND|MPI_CXX_LIBRARIES" build/CMakeCache.txt

# 2. Check if NCCL was found
grep -E "NCCL|nccl" build/CMakeCache.txt | head -20

# 3. Load the necessary modules
module load gcc/12.2.0 cuda/12.4 nvidia/24.3 openmpi/4.1.5-cuda12x-gdr

# 4. Set NCCL environment variables
export NCCL_ROOT=$NVHPC/Linux_x86_64/24.3/comm_libs/nccl
export CMAKE_PREFIX_PATH=$NCCL_ROOT:$CMAKE_PREFIX_PATH
export LD_LIBRARY_PATH=$NCCL_ROOT/lib:$LD_LIBRARY_PATH

# 5. Check if MPI compilers are available
which mpicc
which mpicxx

# 6. Force a clean reconfigure with explicit MPI and NCCL paths
cd ~/alpaka3
cmake -B build \
      -DMPI_C_COMPILER=$(which mpicc) \
      -DMPI_CXX_COMPILER=$(which mpicxx) \
      -DNCCL_ROOT=$NCCL_ROOT \
      -Dalpaka_ENABLE_COLLECTIVES=ON \
      -Dalpaka_ENABLE_NCCL=ON

# 7. Check if MPI and NCCL are now found
grep "MPI_FOUND" build/CMakeCache.txt
grep "NCCL_FOUND" build/CMakeCache.txt

# 8. Rebuild the demo
cmake --build build --target tensorCollectiveDemo -j8

# 9. Verify MPI is now linked
ldd build/example/tensor/collectiveDemo/tensorCollectiveDemo | grep -E "mpi|nccl"


# Set all NCCL and MPI environment variables
export NCCL_ROOT=/trinity/shared/pkg/devel/nvidia/hpc_sdk/Linux_x86_64/24.3/comm_libs/nccl
export NCCL_INCLUDE_DIR=$NCCL_ROOT/include
export NCCL_LIB_DIR=$NCCL_ROOT/lib
export LD_LIBRARY_PATH=$NCCL_ROOT/lib:$LD_LIBRARY_PATH

# NCCL debug and communication settings
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=5
export NCCL_P2P_LEVEL=NVL

# MPI settings
export OMPI_MCA_btl=^openib
export OMPI_MCA_pml=ucx
export OMPI_MCA_osc=ucx



Trouble Shooting

# Check if you still have an allocation
squeue -u $USER

# If not, get a new one
salloc -N2 -n4 --partition=casus_a100 --time=00:30:00

# Once in allocation, run on compute nodes
srun -n 1 nvidia-smi -L

#possible answer after salloc
 srun -n 1 nvidia-smi -L
srun: Warning: can't run 1 processes on 2 nodes, setting nnodes to 1
GPU 0: NVIDIA A100-SXM4-80GB (UUID: GPU-7ac1b215-3501-4e93-836b-e8edd83c18e2)
GPU 1: NVIDIA A100-SXM4-80GB (UUID: GPU-889c4ba4-60de-cd0d-a11e-c434a2adb42e)

# Check your allocation details
scontrol show job $SLURM_JOB_ID

# Check what nodes you have
srun hostname

# Check if CUDA module needs to be loaded on compute nodes
srun -n 1 sh -c 'module list 2>&1 | grep -i cuda'

# Try loading CUDA module on compute nodes
srun -n 1 sh -c 'module load cuda/12.4 && nvidia-smi -L'

# Check the partition
sinfo -p casus_a100


#MPI missing 

module load cuda/12.4 openmpi/4.1.5-cuda12x-gdr gcc/12.2.0
cmake -S .. -B . -DCMAKE_BUILD_TYPE=Release \
      -DMPI_C_COMPILER=$(which mpicc) \
      -DMPI_CXX_COMPILER=$(which mpicxx)
cmake --build . --target tensorCollectiveDemo -j8

grep MPI_ CMakeCache.txt
ldd example/tensor/collectiveDemo/tensorCollectiveDemo | grep mpi
mpirun --oversubscribe -n 4 --map-by ppr:2:node --bind-to none \
       -x NCCL_ROOT -x LD_LIBRARY_PATH \
       ./example/tensor/collectiveDemo/tensorCollectiveDemo
