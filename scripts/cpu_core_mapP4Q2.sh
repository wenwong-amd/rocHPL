#!/bin/bash
# set -x #echo on
set -eu
set -o pipefail

# In this config,
# rank 0 manages GCD 0
# rank 1 manages GCD 2
# rank 2 manages GCD 4
# rank 3 manages GCD 6
# rank 4 manages GCD 1
# rank 5 manages GCD 3
# rank 6 manages GCD 5
# rank 7 manages GCD 7

# GCD0 cores="48-55"
# GCD1 cores="56-63"
# GCD2 cores="16-23"
# GCD3 cores="24-31"
# GCD4 cores="0-7"
# GCD5 cores="8-15"
# GCD6 cores="32-39"
# GCD7 cores="40-47"

set +u
if [[ -n "${OMPI_COMM_WORLD_RANK}" ]]; then
  rank="$OMPI_COMM_WORLD_RANK"
elif [[ -n "${SLURM_PROCID}" ]]; then
  rank="$SLURM_PROCID"
elif [[ -n "${PMI_RANK}" ]]; then
  rank="$PMI_RANK"
fi
set -u

if   [ $rank -eq 0 ]; then
  export OMP_PLACES={48}:8:1,{57}:7:1
elif [ $rank -eq 1 ]; then
  export OMP_PLACES={16}:8:1,{25}:7:1
elif [ $rank -eq 2 ]; then
  export OMP_PLACES={0}:8:1,{9}:7:1
elif [ $rank -eq 3 ]; then
  export OMP_PLACES={32}:8:1,{41}:7:1
elif [ $rank -eq 4 ]; then
  export OMP_PLACES={56}:8:1,{49}:7:1
elif [ $rank -eq 5 ]; then
  export OMP_PLACES={24}:8:1,{17}:7:1
elif [ $rank -eq 6 ]; then
  export OMP_PLACES={8}:8:1,{1}:7:1
elif [ $rank -eq 7 ]; then
  export OMP_PLACES={40}:8:1,{33}:7:1
fi

# export OMP_PLACES=cores
export OMP_PROC_BIND=true

export OMP_NUM_THREADS=15

"$@"

#
# Noel Chalmers
# AMD Research
#

