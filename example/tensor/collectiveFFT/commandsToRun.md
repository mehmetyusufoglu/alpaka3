# Hemera Run Checklist

1. **Load toolchain modules (login node)**
   ```bash
   module load gcc/12.2.0 cuda/12.4 nvidia/24.3 openmpi/4.1.5-cuda12x-gdr
   export NCCL_ROOT=$NVHPC/Linux_x86_64/24.3/comm_libs/nccl
   export LD_LIBRARY_PATH=$NCCL_ROOT/lib:$LD_LIBRARY_PATH
   ```
2. **Request a multi-GPU allocation**
   - `-N` sets how many nodes to allocate (e.g., `-N2` for two nodes).
   - `-n` is the total MPI task count (usually GPUs × nodes when running one rank per GPU).
   - `--gres=gpu:<count>` requests generic GPU resources per node (e.g., `--gres=gpu:2` for two GPUs per node).
   ```bash
   salloc -N2 -n4 --partition=casus_a100 --gres=gpu:2 --time=00:30:00
   # For more resources : salloc -N2 -n8 --partition=casus_a100 --gres=gpu:4 --time=00:3
   # For more memory per GPU --mem can be used: salloc --nodes=1 --gres=gpu:4 --mem=128G
   srun --jobid=$SLURM_JOB_ID --pty bash -l
   ```
3. **Build the demo on the compute node**
   ```bash
   module load cuda/12.4
   cd ~/alpaka3
   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -Dalpaka_ENABLE_COLLECTIVES=ON -Dalpaka_ENABLE_NCCL=ON
   cmake --build build --target tensorCollectiveFFT -j8
   # prepare hostfile for 2 slots 
   scontrol show hostnames "$SLURM_JOB_NODELIST" | awk '{print $0 " slots=2"}' > hostfile
   exit
   ```
4. **Launch the demo from the build tree (login node)**
   ```bash
   cd ~/alpaka3/build
   mpirun -n 4 --hostfile hostfile --map-by ppr:2:node --bind-to none --oversubscribe \
          -x NCCL_ROOT -x LD_LIBRARY_PATH \
          ./example/tensor/collectiveFFT/tensorCollectiveFFT --signal-length=2048 \
          --disable-provider-fft
   ```
   _Note: `--disable-provider-fft` uses host-side DFT until cuFFT provider path is repaired._

   If there are tolerance related error-limit errors. 
   ```bash
   cd ~/alpaka3/build
   mpirun -n 4 --hostfile hostfile --map-by ppr:2:node --bind-to none --oversubscribe \
       -x NCCL_ROOT -x LD_LIBRARY_PATH \
       ./example/tensor/collectiveFFT/tensorCollectiveFFT \
       --signal-length=2048 \
       --verify-abs=1e-1 --verify-rel=5e-3 \
       --verify-direct-precision=float
    ```
   Very large data, skip verify, 2 nodes, 4 gpus
   ```bash
   cd ~/alpaka3/build
mpirun -n 4 --hostfile hostfile --map-by ppr:2:node --bind-to none --oversubscribe           -x NCCL_ROOT -x LD_LIBRARY_PATH           ./example/tensor/collectiveFFT/tensorCollectiveFFT --signal-length=548222976 --force-provider-fft --skip-verify
 ```

   4 nodes, 8 gpus
   ```bash
   cd ~/alpaka3/build
   mpirun -n 8 --hostfile hostfile --map-by ppr:4:node --bind-to none --oversubscribe \
      -x NCCL_ROOT -x LD_LIBRARY_PATH \
      ./example/tensor/collectiveFFT/tensorCollectiveFFT --signal-length=548222976 \
      --force-provider-fft --skip-verify
   ```
   ⚠️ 548,222,976 complex samples require roughly 8.8 GiB per rank (two full-sized buffers before NCCL),
   so with four ranks per node Hemera's OOM killer will terminate the job unless extra memory nodes are
   requested or the algorithm is refactored to avoid the full-spectrum staging.

   3 nodes, 12 gpus
   ```bash
   cd ~/alpaka3/build
   mpirun -n 12 --hostfile hostfile --map-by ppr:4:node --bind-to none --oversubscribe \
      -x NCCL_ROOT -x LD_LIBRARY_PATH \
      ./example/tensor/collectiveFFT/tensorCollectiveFFT --signal-length=548222976 \
      --force-provider-fft --skip-verify
   ```

     4 nodes, 16 gpus
   ```bash
   cd ~/alpaka3/build
   mpirun -n 16 --hostfile hostfile --map-by ppr:4:node --bind-to none --oversubscribe \
      -x NCCL_ROOT -x LD_LIBRARY_PATH \
      ./example/tensor/collectiveFFT/tensorCollectiveFFT --signal-length=548222976 \
      --force-provider-fft --skip-verify
   ```

6. **Confirm output**
   - Each rank reports its strided sample range.
   - Rank 0 prints a spectrum preview and verification result.
   - "Distributed FFT complete" message indicates success.
make 



## Scaling To Eight GPUs Across Four Nodes

1. **Request a larger allocation (login node)**
   ```bash
   salloc -N4 -n8 --partition=casus_a100 --gres=gpu:2 --time=00:45:00
   srun --jobid=$SLURM_JOB_ID --pty bash -l
   ```
   - `-N4`: reserve four nodes to host two GPUs each.
   - `-n8`: launch eight MPI ranks total (one per GPU).
   - `--partition=casus_a100`: pick the A100-equipped queue.
   - `--gres=gpu:2`: expose two GPUs on every allocated node.
   - `--time=00:45:00`: keep the reservation for 45 minutes.
   - `srun --jobid=$SLURM_JOB_ID --pty bash -l`: open an interactive shell on the compute allocation.
2. **Generate an eight-slot hostfile on the compute node**
   ```bash
   scontrol show hostnames "$SLURM_JOB_NODELIST" | awk '{print $0 " slots=2"}' > hostfile
   ```
3. **Launch the FFT demo using eight ranks (login node)**
   ```bash
   cd ~/alpaka3/build
   mpirun -n 8 --hostfile hostfile --map-by ppr:2:node --bind-to none --oversubscribe \
      -x NCCL_ROOT -x LD_LIBRARY_PATH \
      ./example/tensor/collectiveFFT/tensorCollectiveFFT --signal-length=548222976 \
      --force-provider-fft --skip-verify
   ```
   - `-n 8`: start one MPI process per GPU.
   - `--hostfile hostfile`: reuse the eight-slot hostfile generated on the compute nodes.
   - `--map-by ppr:2:node`: place two ranks on each node (matching two GPUs per node).
   - `--bind-to none`: prevent OpenMPI from pinning ranks to specific CPU cores.
   - `--oversubscribe`: allow OpenMPI to launch even if logical CPU slots appear exhausted.
   - `-x NCCL_ROOT -x LD_LIBRARY_PATH`: forward required NCCL and CUDA library paths to each rank.
   - `--signal-length=548222976`: request the large global FFT size to stay under per-rank cuFFT limits.
   - `--force-provider-fft --skip-verify`: insist on GPU FFT execution and disable O(N^2) verification for large input.

## Memory Allocation for Large Runs

When running very large FFTs (hundreds of millions of samples), the NCCL all-reduce phase can require substantial memory for communication buffers. If ranks are killed with signal 9 during execution, request all available node memory:

```bash
salloc -N4 -n16 --partition=casus_a100 --gres=gpu:4 --mem=0 --time=00:30:00
```

The `--mem=0` flag allocates all available memory on each node, preventing OOM kills during the collective communication phase. This is particularly important when running 4+ ranks per node with signal sizes approaching 2^29 samples.

Alternatively, specify an explicit amount (e.g., `--mem=200G`) if you know the per-node requirement.
