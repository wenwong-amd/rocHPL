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

#define BLOCK_SIZE 256

static __global__ void dlaswp04T(const int M,
                                 const int N,
                                 double* __restrict__ A,
                                 const int LDA,
                                 double* __restrict__ W,
                                 const int LDW,
                                 const int* __restrict__ LINDXU) {

  const int m = blockIdx.x;
  const int n = threadIdx.x + BLOCK_SIZE * blockIdx.y;

  if (n<N) {
    const int ipa  = LINDXU[m];

    //sparse write of A, but coalesced read of W
    A[ipa + n * ((size_t)LDA)] = W[n + m * ((size_t)LDW)];
  }
}

void HPL_dlaswp04T(const int  M,
                   const int  N,
                   double*    A,
                   const int  LDA,
                   double*    W,
                   const int  LDW,
                   const int* LINDXU) {
  /*
   * Purpose
   * =======
   *
   * HPL_dlaswp04T writes columns  of  W  into  rows  of  A  at  positions
   * indicated by LINDXU.
   *
   * Arguments
   * =========
   *
   * M       (local input)                 const int
   *         On entry, M  specifies the number of rows of A that should be
   *         replaced with columns of W. M must be at least zero.
   *
   * N       (local input)                 const int
   *         On entry, N specifies the length of the rows of A that should
   *         be replaced with columns of W. N must be at least zero.
   *
   * A       (local output)                double *
   *         On entry, A points to an array of dimension (LDA,N). On exit,
   *         the  rows of this array specified by  LINDXU  are replaced by
   *         columns of W.
   *
   * LDA     (local input)                 const int
   *         On entry, LDA specifies the leading dimension of the array A.
   *         LDA must be at least MAX(1,M).
   *
   * W       (local input/output)          double *
   *         On entry,  W  points  to an array of dimension (LDW,*).  This
   *         array contains the columns of  W  that are to be writen to
   *         rows of A.
   *
   * LDW     (local input)                 const int
   *         On entry, LDW specifies the leading dimension of the array W.
   *         LDW must be at least MAX(1,N).
   *
   * LINDXU  (local input)                 const int *
   *         On entry, LINDXU is an array of dimension M that contains the
   *         local row indexes of A that should be replaced with W.
   *
   * ---------------------------------------------------------------------
   */
  /*
   * .. Local Variables ..
   */

  if((M <= 0) || (N <= 0)) return;

  dim3 grid_size(M, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
  dim3 block_size(BLOCK_SIZE);
  hipLaunchKernelGGL((dlaswp04T),
                     grid_size,
                     block_size,
                     0,
                     computeStream,
                     M,
                     N,
                     A,
                     LDA,
                     W,
                     LDW,
                     LINDXU);

  /*
   * End of HPL_dlaswp04T
   */
}
