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

hipEvent_t panelCopy, swapDataTransfer, L1Transfer, L2Transfer;
hipEvent_t pdlaswpStart_1, pdlaswpStart_2;
hipEvent_t pdlaswpFinish_1, pdlaswpFinish_2;
hipEvent_t swapStartEvent[HPL_N_UPD], update[HPL_N_UPD];
hipEvent_t swapUCopyEvent[HPL_N_UPD], swapWCopyEvent[HPL_N_UPD];
hipEvent_t dgemmStart[HPL_N_UPD], dgemmStop[HPL_N_UPD];

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

  hipEventCreate(&panelCopy);
  hipEventCreate(&swapDataTransfer);
  hipEventCreate(&L1Transfer);
  hipEventCreate(&L2Transfer);

  hipEventCreate(&pdlaswpStart_1);
  hipEventCreate(&pdlaswpStart_2);
  hipEventCreate(&pdlaswpFinish_1);
  hipEventCreate(&pdlaswpFinish_2);

  hipEventCreate(swapStartEvent + HPL_LOOK_AHEAD);
  hipEventCreate(swapStartEvent + HPL_UPD_1);
  hipEventCreate(swapStartEvent + HPL_UPD_2);

  hipEventCreate(swapUCopyEvent + HPL_LOOK_AHEAD);
  hipEventCreate(swapUCopyEvent + HPL_UPD_1);
  hipEventCreate(swapUCopyEvent + HPL_UPD_2);

  hipEventCreate(swapWCopyEvent + HPL_LOOK_AHEAD);
  hipEventCreate(swapWCopyEvent + HPL_UPD_1);
  hipEventCreate(swapWCopyEvent + HPL_UPD_2);

  hipEventCreate(update + HPL_LOOK_AHEAD);
  hipEventCreate(update + HPL_UPD_1);
  hipEventCreate(update + HPL_UPD_2);

  hipEventCreate(dgemmStart + HPL_LOOK_AHEAD);
  hipEventCreate(dgemmStart + HPL_UPD_1);
  hipEventCreate(dgemmStart + HPL_UPD_2);

  hipEventCreate(dgemmStop + HPL_LOOK_AHEAD);
  hipEventCreate(dgemmStop + HPL_UPD_1);
  hipEventCreate(dgemmStop + HPL_UPD_2);
}

void Free_gpu() {
  rocblas_destroy_handle(handle);

  hipStreamDestroy(computeStream);
  hipStreamDestroy(dataStream);
}
