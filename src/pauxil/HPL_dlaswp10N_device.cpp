/* ---------------------------------------------------------------------
 * -- High Performance Computing Linpack Benchmark (HPL)
 *    HPL - 2.2 - February 24, 2016
 *    Antoine P. Petitet
 *    University of Tennessee, Knoxville
 *    Innovative Computing Laboratory
 *    (C) Copyright 2000-2008 All Rights Reserved
 *
 *    Modified by: Noel Chalmers
 *    (C) 2018-2021 Advanced Micro Devices, Inc.
 *    See the rocHPL/LICENCE file for details.
 *
 *    SPDX-License-Identifier: (BSD-3-Clause)
 * ---------------------------------------------------------------------
 */

#include "hpl.hpp"
#include <hip/hip_runtime.h>
#include <cassert>

#define assertm(exp, msg) assert(((void)msg, exp))

#define MAX_SWAP 1024
#define BLOCK_SIZE 256

__global__ void dlaswp10N(const int M,
                          const int N,
                          double* __restrict__ A,
                          const int LDA,
                          const int* __restrict__ IPIV) {

  __shared__ int s_ipiv[MAX_SWAP];

  // Load Pivots
  for (int i=threadIdx.x;i<N;i+=blockDim.x)
    s_ipiv[i] = IPIV[i];

  __syncthreads();

  const int m = threadIdx.x + BLOCK_SIZE * blockIdx.x;

  for(int i = 0; i < N; i++) {
    int ip = s_ipiv[i];

    if (ip == i) continue;

    // Read src column
    double Ai;
    if (m<M) Ai = A[m + i * ((size_t)LDA)];

    while(ip != i) {
      // Save dst column to reg
      double Aip;
      if (m<M) Aip = A[m + ip * ((size_t)LDA)];

      // Write Ai into Aip
      if (m<M) A[m + ip * ((size_t)LDA)] = Ai;

      Ai = Aip;
      // Read where the new Ai should go
      const int ip_next = s_ipiv[ip];
      __syncthreads();

      // Update the s_ipiv[ip] location with thread 0
      if (threadIdx.x==0) s_ipiv[ip] = ip;
      __syncthreads();

      ip = ip_next;
    }

    // After the while loop, Ai contains the column for index i
    if (m<M) A[m + i * ((size_t)LDA)] = Ai;
  }
}

void HPL_dlaswp10N(const int  M,
                   const int  N,
                   double*    A,
                   const int  LDA,
                   const int* IPIV) {
  /*
   * Purpose
   * =======
   *
   * HPL_dlaswp10N performs a sequence  of  local column interchanges on a
   * matrix A.  One column interchange is initiated  for columns 0 through
   * N-1 of A.
   *
   * Arguments
   * =========
   *
   * M       (local input)                 const int
   *         __arg0__
   *
   * N       (local input)                 const int
   *         On entry,  M  specifies  the number of rows of the array A. M
   *         must be at least zero.
   *
   * A       (local input/output)          double *
   *         On entry, N specifies the number of columns of the array A. N
   *         must be at least zero.
   *
   * LDA     (local input)                 const int
   *         On entry, A  points to an  array of  dimension (LDA,N).  This
   *         array contains the columns onto which the interchanges should
   *         be applied. On exit, A contains the permuted matrix.
   *
   * IPIV    (local input)                 const int *
   *         On entry, LDA specifies the leading dimension of the array A.
   *         LDA must be at least MAX(1,M).
   *
   * ---------------------------------------------------------------------
   */

  if((M <= 0) || (N <= 0)) return;

  assertm(N <= 1024, "NB too large in HPL_dlaswp10N");

  hipStream_t stream;
  rocblas_get_stream(handle, &stream);

  dim3 grid_size((M + BLOCK_SIZE - 1) / BLOCK_SIZE);
  hipLaunchKernelGGL((dlaswp10N),
                     grid_size,
                     dim3(BLOCK_SIZE),
                     0,
                     stream,
                     M,
                     N,
                     A,
                     LDA,
                     IPIV);
}
