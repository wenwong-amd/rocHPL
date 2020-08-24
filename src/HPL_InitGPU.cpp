/*
 * Copyright (c) 2018-2020 Advanced Micro Devices, Inc.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holder nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 * without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include "hpl.h"

rocblas_handle handle;

hipStream_t computeStream, dataStream;

hipEvent_t panelUpdate;
hipEvent_t panelCopy;

hipEvent_t dlaswpStart, dlaswpStop;
hipEvent_t dtrsmStart, dtrsmStop;
hipEvent_t dgemmStart, dgemmStop;

int stringCmp(const void* a, const void* b) {
  char* c_a = (char*)a;
  char* c_b = (char*)b;
  return strcmp(c_a, c_b);
}

static char host_name[MPI_MAX_PROCESSOR_NAME];

/*
  This function finds out how many MPI processes are running on the same node
  and assigns a local rank that can be used to map a process to a device.
  This function needs to be called by all the MPI processes.
*/
void HPL_InitGPU() {
  char(*host_names)[MPI_MAX_PROCESSOR_NAME];

  int    i, n, namelen, color, rank, nprocs;
  size_t bytes;
  int    dev;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Get_processor_name(host_name, &namelen);

  bytes      = nprocs * sizeof(char[MPI_MAX_PROCESSOR_NAME]);
  host_names = (char(*)[MPI_MAX_PROCESSOR_NAME])malloc(bytes);

  strcpy(host_names[rank], host_name);

  for(n = 0; n < nprocs; n++) {
    MPI_Bcast(
        &(host_names[n]), MPI_MAX_PROCESSOR_NAME, MPI_CHAR, n, MPI_COMM_WORLD);
  }

  int localRank = 0;
  for(n = 0; n < rank; n++) {
    if(!strcmp(host_name, host_names[n])) localRank++;
  }

  int localSize = 0;
  for(n = 0; n < nprocs; n++) {
    if(!strcmp(host_name, host_names[n])) localSize++;
  }

  /* Find out how many DP capable GPUs are in the system and their device number
   */
  int deviceCount;
  hipGetDeviceCount(&deviceCount);

#ifdef VERBOSE_PRINT
  printf("Assigning device %d on node %s to rank %d \n",
         localRank % deviceCount,
         host_name,
         rank);
#endif

  /* Assign device to MPI process, initialize BLAS and probe device properties
   */
  dev = localRank % deviceCount;
  hipSetDevice(dev);

  rocblas_create_handle(&handle);
  rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);

  rocblas_initialize();

  hipStreamCreate(&computeStream);
  hipStreamCreate(&dataStream);

  rocblas_set_stream(handle, computeStream);

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
