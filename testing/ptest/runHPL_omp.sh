#!/bin/bash

# FOR OMP
export OMP_NUM_THREADS=16
# export BLIS_NUM_THREADS=8

export LD_LIBRARY_PATH=~/openblas:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=~/amd-blis-mt-1.2/lib:$LD_LIBRARY_PATH

# ./xhpl
#HSA_ENABLE_SDMA=1 mpiexec -np 1 bin/ROCM/xhpl
HSA_ENABLE_SDMA=1 hwloc-bind --physical --cpubind package:0.pu:0-15 ./bin/ROCM/xhpl
