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
   # For more resources : salloc -N2 -n8 --partition=casus_a100 --gres=gpu:4 --time=00:30:00
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
          ./example/tensor/collectiveFFT/tensorCollectiveFFT --signal-length=65536
   ```

   If there are tolerance related error-limit errors. 
   ```bash
   cd ~/alpaka3/build
   mpirun -n 4 --hostfile hostfile --map-by ppr:2:node --bind-to none --oversubscribe \
       -x NCCL_ROOT -x LD_LIBRARY_PATH \
       ./example/tensor/collectiveFFT/tensorCollectiveFFT \
       --signal-length=65536 \
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
   ⚠️ 548,222,976 complex samples require roughly 8.8 GiB per rank (two full-sized buffers before NCCL),
   so with four ranks per node Hemera's OOM killer will terminate the job unless extra memory nodes are
   requested or the algorithm is refactored to avoid the full-spectrum staging.
6. **Confirm output**
   - Each rank reports its strided sample range.
   - Rank 0 prints a spectrum preview and verification result.
   - "Distributed FFT complete" message indicates success.
make 
