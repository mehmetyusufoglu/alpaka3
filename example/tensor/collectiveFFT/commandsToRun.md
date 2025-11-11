# Hemera Run Checklist

1. **Load toolchain modules (login node)**
   ```bash
   module load gcc/12.2.0 cuda/12.4 nvidia/24.3 openmpi/4.1.5-cuda12x-gdr
   export NCCL_ROOT=$NVHPC/Linux_x86_64/24.3/comm_libs/nccl
   export LD_LIBRARY_PATH=$NCCL_ROOT/lib:$LD_LIBRARY_PATH
   ```
2. **Build the demo**
   ```bash
   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -Dalpaka_ENABLE_COLLECTIVES=ON -Dalpaka_ENABLE_NCCL=ON
   cmake --build build --target tensorCollectiveFFT -j8
   ```
3. **Request a multi-GPU allocation**
   ```bash
   salloc -N2 -n4 --partition=casus_a100 --gres=gpu:2 --time=00:30:00
   srun --jobid=$SLURM_JOB_ID --pty bash -l
   ```
4. **Prepare hostfile on compute node**
   ```bash
   module load cuda/12.4
   scontrol show hostnames "$SLURM_JOB_NODELIST" | awk '{print $0 " slots=2"}' > hostfile
   exit
   ```
5. **Launch the demo from the build tree (login node)**
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
6. **Confirm output**
   - Each rank reports its strided sample range.
   - Rank 0 prints a spectrum preview and verification result.
   - "Distributed FFT complete" message indicates success.
