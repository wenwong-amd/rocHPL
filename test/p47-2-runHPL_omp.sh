#!/bin/bash

num_cpu_cores=8
num_gpus=4

MPI_DIR=../local/openmpi

# FOR OMP
export OMP_NUM_THREADS=${num_cpu_cores}

# run with openmpi + ucx
HSA_ENABLE_SDMA=1 sudo ${MPI_DIR}/bin/mpirun -np ${num_gpus} --allow-run-as-root -x UCX_TLS=sm,rocm_ipc,rocm_cpy --mca pml ucx --mca btl ^vader,tcp,openib,uct --map-by socket:PE=${num_cpu_cores} --bind-to core:overload-allowed --report-bindings ./bin/ROCM_UCX/xhpl
