#!/bin/bash

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
