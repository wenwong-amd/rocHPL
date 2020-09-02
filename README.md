# rocHPL
rocHPL is a benchmark based on the [HPL][] benchmark application, implemented on top of AMD's Radeon Open Compute [ROCm][] Platform, runtime, and toolchains. rocHPL is created using the [HIP][] programming language and optimized for AMD's latest discrete GPUs.

## Requirements
* Git
* CMake (3.10 or later)
* MPI
* NUMA library
* AMD [ROCm] platform (3.5 or later)
* [rocBLAS][]
* [rocRAND][]

## Quickstart rocHPL build and install

#### Install script
You can build rocHPL using the *install.sh* script
```
# Clone rocHPL using git
git clone https://github.com/ROCmSoftwarePlatform/rocHPL.git

# Go to rocHPL directory
cd rocHPL

# Run install.sh script
# Command line options:
#    -h|--help            - prints this help message
#    -i|--install]        - install after build
#    -d|--dependencies    - install dependencies
#    -g|--debug]          - Set build type to Debug (otherwise build Release)
#    --with-cpublas=<dir> - Path to external CPU BLAS library (Default: clone+build OpenBLAS)
#    --with-mpi=<dir>     - Path to external MPI install (Default: clone+build OpenMPI)
#    --gpu-aware-mpi      - MPI library supports GPU-aware communication (Default: false)
#    --verbose-print      - Verbose output during HPL setup (Default: true)
#    --progress-report    - Print progress report to terminal during HPL run (Default: true)
#    --detailed-timing    - Record detailed timers during HPL run (Default: true)
./install.sh -di
```
By default, [OpenBLAS] v0.3.10, [UCX] v1.8.1, and [OpenMPI] v4.0.5 will be cloned and build in rocHPL/tpl. After install and build, the `rochpl` executable is placed in build/rochpl-install.

## Running rocHPL benchmark application
You can run the rocHPL benchmark application by running the executable with MPI directly, or by using a provided `run_rochpl` script configured at build
```
run_rochpl --np <np> --ppn <ppn>
# where
# np      - is the number of MPI ranks to run with
# ppn     - is the number of ranks per node you are running (important for CPU partitioning)
```

The `rochpl` uses an input file `HPL.dat`, e.g.
```
HPLinpack benchmark input file
Innovative Computing Laboratory, University of Tennessee
HPL.out      output file name (if any)
0            device out (6=stdout,7=stderr,file)
1            # of problems sizes (N)
40000        Ns
1            # of NBs
384          NBs
0            PMAP process mapping (0=Row-,1=Column-major)
1            # of process grids (P x Q)
1            Ps
1            Qs
16.0         threshold
1            # of panel fact
2            PFACTs (0=left, 1=Crout, 2=Right)
1            # of recursive stopping criterium
2            NBMINs (>= 1)
1            # of panels in recursion
2            NDIVs
1            # of recursive panel fact.
2            RFACTs (0=left, 1=Crout, 2=Right)
1            # of broadcast
6            BCASTs (0=1rg,1=1rM,2=2rg,3=2rM,4=Lng,5=LnM,6=Ibcast)
1            # of lookahead depth
1            DEPTHs (>=0)
1            SWAP (0=bin-exch,1=long,2=mix)
64           swapping threshold
1            L1 in (0=transposed,1=no-transposed) form
0            U  in (0=transposed,1=no-transposed) form
0            Equilibration (0=no,1=yes)
8            memory alignment in double (> 0)
```

## Performance evaluation


## Testing rocHPL

Please note that for successful testing, a device with at least 16GB of device memory is required.

## Support
Please use [the issue tracker][] for bugs and feature requests.

## License
The [license file][] can be found in the main repository.

[HPL]: http://icl.utk.edu/hpl/
[ROCm]: https://github.com/RadeonOpenCompute/ROCm
[HIP]: https://github.com/ROCm-Developer-Tools/HIP
[rocBLAS]: https://github.com/ROCmSoftwarePlatform/rocBLAS
[rocRAND]: https://github.com/ROCmSoftwarePlatform/rocRAND
[OpenBLAS]: https://github.com/xianyi/OpenBLAS
[OpenMPI]: https://github.com/open-mpi/ompi
[UCX]: https://github.com/openucx/ucx
[the issue tracker]: https://github.com/ROCmSoftwarePlatform/rocHPL/issues
[license file]: https://github.com/ROCmSoftwarePlatform/rocHPL
