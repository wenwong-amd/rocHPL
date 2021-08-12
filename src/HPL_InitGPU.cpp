/* ---------------------------------------------------------------------
 * -- High Performance Computing Linpack Benchmark (HPL)
 *    Noel Chalmers
 *    (C) 2018-2021 Advanced Micro Devices, Inc.
 *    See the rocHPL/LICENCE file for details.
 *
 *    SPDX-License-Identifier: (BSD-3-Clause)
 * ---------------------------------------------------------------------
 */

#include "hpl.hpp"
#include <algorithm>

rocblas_handle handle;

hipStream_t computeStream, dataStream;

hipEvent_t panelUpdate;
hipEvent_t panelCopy;

hipEvent_t swapStartEvent, swapUCopyEvent, swapWCopyEvent;

hipEvent_t dlaswpStart, dlaswpStop;
hipEvent_t dtrsmStart, dtrsmStop;
hipEvent_t dgemmStart, dgemmStop;

static char host_name[MPI_MAX_PROCESSOR_NAME];

/*
  This function finds out how many MPI processes are running on the same node
  and assigns a local rank that can be used to map a process to a device.
  This function needs to be called by all the MPI processes.
*/
void HPL_InitGPU(const HPL_T_grid* GRID) {
  char host_name[MPI_MAX_PROCESSOR_NAME];

  int i, n, namelen, rank, nprocs;
  int dev;

  int nprow, npcol, myrow, mycol;
  (void)HPL_grid_info(GRID, &nprow, &npcol, &myrow, &mycol);

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  MPI_Get_processor_name(host_name, &namelen);

  MPI_Comm nodeComm;
  MPI_Comm_split_type(
      MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);

  int localRank;
  int localSize;
  MPI_Comm_rank(nodeComm, &localRank);
  MPI_Comm_size(nodeComm, &localSize);

  /* Find out how many GPUs are in the system and their device number */
  int deviceCount;
  hipGetDeviceCount(&deviceCount);

  if (deviceCount<1) {
    if(localRank == 0)
      HPL_pwarn(stderr,
                __LINE__,
                "HPL_InitGPU",
                "Node %s found no GPUs. Is the ROCm kernel module loaded?",
                host_name);
    MPI_Finalize();
    exit(1);
  }

  typedef struct {
    int p;
    int q;
  } int2;

  int2 mypq{myrow, mycol};
  int2 pq[localSize];
  MPI_Allgather(&mypq, 2, MPI_INT, pq, 2, MPI_INT, nodeComm);

  // sort by P then by Q
  std::sort(pq, pq + localSize, [](const int2& a, const int2& b) {
    if(a.p < b.p) return true;
    if(a.p > b.p) return false;

    return (a.q < b.q);
  });

  for(int i = 0; i < localSize; ++i) {
    if(pq[i].p == mypq.p && pq[i].q == mypq.q) {
      dev = i;
      break;
    }
  }

  dev = dev % deviceCount;

  MPI_Comm_free(&nodeComm);

#ifdef HPL_VERBOSE_PRINT
  hipDeviceProp_t props;
  hipGetDeviceProperties(&props, dev);

  printf("Assigning device %d, pciBusID %x, on node %s to rank %d \n",
         dev,
         props.pciBusID,
         host_name,
         rank);
#endif

  /* Assign device to MPI process, initialize BLAS and probe device properties
   */
  hipSetDevice(dev);

  hipStreamCreate(&computeStream);
  hipStreamCreate(&dataStream);

  rocblas_create_handle(&handle);
  rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);

  rocblas_initialize();

  rocblas_set_stream(handle, computeStream);

  hipEventCreate(&swapStartEvent);
  hipEventCreate(&swapUCopyEvent);
  hipEventCreate(&swapWCopyEvent);

  hipEventCreate(&panelUpdate);
  hipEventCreate(&panelCopy);
  hipEventCreate(&dlaswpStart);
  hipEventCreate(&dlaswpStop);
  hipEventCreate(&dtrsmStart);
  hipEventCreate(&dtrsmStop);
  hipEventCreate(&dgemmStart);
  hipEventCreate(&dgemmStop);
}

void Free_gpu() {
  rocblas_destroy_handle(handle);

  hipStreamDestroy(computeStream);
  hipStreamDestroy(dataStream);
}
