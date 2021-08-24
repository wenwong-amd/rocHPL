#!/bin/bash
# set -x #echo on
set -eu
set -o pipefail

# cores[0]="48-55"
# cores[1]="56-63"
# cores[2]="16-23"
# cores[3]="24-31"
# cores[4]="0-7"
# cores[5]="8-15"
# cores[6]="32-39"
# cores[7]="40-47"

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
  export OMP_PLACES={48}:8:1,{32}:8:1,{40}:8:1,{57}:7:1
elif [ $rank -eq 1 ]; then
  export OMP_PLACES={16}:8:1,{0}:8:1,{8}:8:1,{25}:7:1
elif [ $rank -eq 2 ]; then
  export OMP_PLACES={56}:8:1,{32}:8:1,{40}:8:1,{49}:7:1
elif [ $rank -eq 3 ]; then
  export OMP_PLACES={24}:8:1,{0}:8:1,{8}:8:1,{17}:7:1
fi

# export OMP_PLACES=cores
export OMP_PROC_BIND=true

export OMP_NUM_THREADS=31

"$@"

#
# Noel Chalmers
# AMD Research
#

