/* ---------------------------------------------------------------------
 * -- High Performance Computing Linpack Benchmark (HPL)
 *    HPL - 2.2 - February 24, 2016
 *    Antoine P. Petitet
 *    University of Tennessee, Knoxville
 *    Innovative Computing Laboratory
 *    (C) Copyright 2000-2008 All Rights Reserved
 *
 *    Modified by: Noel Chalmers
 *    (C) 2018-2020 Advanced Micro Devices, Inc.
 *    See the rocHPL/LICENCE file for details.
 *
 *    SPDX-License-Identifier: (BSD-3-Clause)
 * ---------------------------------------------------------------------
 */

#include "hpl.hpp"
#include <hip/hip_runtime.h>
#include <cassert>

#define assertm(exp, msg) assert(((void)msg, exp))

#define BLOCK_SIZE 1024

/* Perform any local row swaps of A */
__global__ void dlaswp02T(const int M,
                          const int N,
                          double* __restrict__ A,
                          const int LDA,
                          const int* __restrict__ LINDXA,
                          const int* __restrict__ LINDXAU) {

  __shared__ double s_A[BLOCK_SIZE];

  const int n = blockIdx.x;
  const int m = threadIdx.x;

  int ipau, ipa;

  if(m < M) {
    ipau = LINDXAU[m];
    ipa  = LINDXA[m];

    // read in
    s_A[m] = (ipau < 0) ? A[ipa + n * ((size_t)LDA)] : 0.0;
  }
  __syncthreads();

  if(m < M) {
    if(ipau < 0) { // swap into A
      A[-ipau + n * ((size_t)LDA)] = s_A[m];
    }
  }
}

void HPL_dlaswp02T(const int  M,
                   const int  N,
                   double*    A,
                   const int  LDA,
                   const int* LINDXA,
                   const int* LINDXAU) {
  /*
   * Purpose
   * =======
   *
   * HPL_dlaswp02T copies  scattered rows  of  A  into itself. The row
   * offsets in  A  of the source rows  are specified by LINDXA.
   * The  destination of those rows are specified by  LINDXAU.  A
   * positive value of LINDXAU indicates that the array  destination is U,
   * and A otherwise.
   *
   * Arguments
   * =========
   *
   * M       (local input)                 const int
   *         On entry, M  specifies the number of rows of A that should be
   *         moved within A or copied into U. M must be at least zero.
   *
   * N       (local input)                 const int
   *         On entry, N  specifies the length of rows of A that should be
   *         moved within A or copied into U. N must be at least zero.
   *
   * A       (local input/output)          double *
   *         On entry, A points to an array of dimension (LDA,N). The rows
   *         of this array specified by LINDXA should be moved within A or
   *         copied into U.
   *
   * LDA     (local input)                 const int
   *         On entry, LDA specifies the leading dimension of the array A.
   *         LDA must be at least MAX(1,M).
   *
   * LINDXA  (local input)                 const int *
   *         On entry, LINDXA is an array of dimension M that contains the
   *         local  row indexes  of  A  that should be moved within  A  or
   *         or copied into U.
   *
   * LINDXAU (local input)                 const int *
   *         On entry, LINDXAU  is an array of dimension  M that  contains
   *         the local  row indexes of  U  where the rows of  A  should be
   *         copied at. This array also contains the  local row offsets in
   *         A where some of the rows of A should be moved to.  A positive
   *         value of  LINDXAU[i]  indicates that the row  LINDXA[i]  of A
   *         should be copied into U at the position LINDXAU[i]; otherwise
   *         the row  LINDXA[i]  of  A  should be moved  at  the  position
   *         -LINDXAU[i] within A.
   *
   * ---------------------------------------------------------------------
   */
  /*
   * .. Local Variables ..
   */

  if((M <= 0) || (N <= 0)) return;

  assertm(M <= BLOCK_SIZE, "NB too large in HPL_dlaswp02T");

  hipLaunchKernelGGL(
      (dlaswp02T), N, M, 0, computeStream, M, N, A, LDA, LINDXA, LINDXAU);

  /*
   * End of HPL_dlaswp02T
   */
}
