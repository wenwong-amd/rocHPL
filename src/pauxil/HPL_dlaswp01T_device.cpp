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

/* Build U matrix from rows of A */
__global__ void dlaswp01T(const int M,
                          const int N,
                          double* __restrict__ A,
                          const int LDA,
                          double* __restrict__ U,
                          const int LDU,
                          const int* __restrict__ LINDXU) {

  const int m = blockIdx.x;
  const int n = threadIdx.x + BLOCK_SIZE * blockIdx.y;

  if (n<N) {
    const int ipa  = LINDXU[m];

    //sparse access of A, but coalesced write to U
    U[n + m * ((size_t)LDU)] = A[ipa + n * ((size_t)LDA)];
  }
}

void HPL_dlaswp01T(const int  M,
                   const int  N,
                   double*    A,
                   const int  LDA,
                   double*    U,
                   const int  LDU,
                   const int* LINDXU) {
  /*
   * Purpose
   * =======
   *
   * HPL_dlaswp01T copies  scattered rows  of  A  into an array U.  The
   * row offsets in  A  of the source rows  are specified by LINDXU.
   * Rows of A are stored as columns in U.
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
   * U       (local input/output)          double *
   *         On entry, U points to an array of dimension (LDU,M). The rows
   *         of A specified by  LINDXA  are copied within this array  U at
   *         the  positions indicated by positive values of LINDXAU.  The
   *         rows of A are stored as columns in U.
   *
   * LDU     (local input)                 const int
   *         On entry, LDU specifies the leading dimension of the array U.
   *         LDU must be at least MAX(1,N).
   *
   * LINDXU  (local input)                 const int *
   *         On entry, LINDXU is an array of dimension M that contains the
   *         local  row indexes  of  A  that should be copied into U.
   *
   * ---------------------------------------------------------------------
   */
  /*
   * .. Local Variables ..
   */

  if((M <= 0) || (N <= 0)) return;

  dim3 grid_size(M, (N+BLOCK_SIZE-1)/BLOCK_SIZE );
  dim3 block_size(BLOCK_SIZE);
  hipLaunchKernelGGL((dlaswp01T),
                     grid_size,
                     block_size,
                     0,
                     computeStream,
                     M,
                     N,
                     A,
                     LDA,
                     U,
                     LDU,
                     LINDXU);

  /*
   * End of HPL_dlaswp01T
   */
}
