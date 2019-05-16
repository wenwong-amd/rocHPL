#!/bin/bash

num_cpu_cores=32
num_gpus=1

MPI_DIR=/usr/local/openmpi

# FOR OMP
export OMP_NUM_THREADS=${num_cpu_cores}
export LD_LIBRARY_PATH=openblas:$LD_LIBRARY_PATH

# ./xhpl
HSA_ENABLE_SDMA=1 ${MPI_DIR}/bin/mpirun -np ${num_gpus} --map-by socket:PE=${num_cpu_cores} --bind-to core:overload-allowed --report-bindings ./bin/ROCM/xhpl
