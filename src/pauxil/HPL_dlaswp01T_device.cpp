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

#define TILE_DIM 32
#define BLOCK_ROWS 8

#define assertm(exp, msg) assert(((void)msg, exp))

/* Build U matrix from rows of A */
__global__ void dlaswp01T(const int M,
                          const int N,
                          double* __restrict__ A,
                          const int LDA,
                          double* __restrict__ U,
                          const int LDU,
                          const int* __restrict__ LINDXA,
                          const int* __restrict__ LINDXAU) {

  __shared__ double s_U[TILE_DIM][TILE_DIM + 1];

  const int m = threadIdx.x + TILE_DIM * blockIdx.x;
  const int n = threadIdx.y + TILE_DIM * blockIdx.y;

  if(m < M) {
    const int ipa  = LINDXA[m];
    const int ipau = LINDXAU[m];

    if(ipau >= 0) { // row will swap into U
      // save in LDS for the moment
      // possible cache-hits if ipas are close
      s_U[threadIdx.x][threadIdx.y + 0] =
          (n + 0 < N) ? A[ipa + (n + 0) * ((size_t)LDA)] : 0.0;
      s_U[threadIdx.x][threadIdx.y + 8] =
          (n + 8 < N) ? A[ipa + (n + 8) * ((size_t)LDA)] : 0.0;
      s_U[threadIdx.x][threadIdx.y + 16] =
          (n + 16 < N) ? A[ipa + (n + 16) * ((size_t)LDA)] : 0.0;
      s_U[threadIdx.x][threadIdx.y + 24] =
          (n + 24 < N) ? A[ipa + (n + 24) * ((size_t)LDA)] : 0.0;
    }
  }

  __syncthreads();

  const int um = threadIdx.y + TILE_DIM * blockIdx.x;
  const int un = threadIdx.x + TILE_DIM * blockIdx.y;

  if(un < N) {
    const int uipau0 = (um + 0 < M) ? LINDXAU[um + 0] : -1;
    const int uipau1 = (um + 8 < M) ? LINDXAU[um + 8] : -1;
    const int uipau2 = (um + 16 < M) ? LINDXAU[um + 16] : -1;
    const int uipau3 = (um + 24 < M) ? LINDXAU[um + 24] : -1;

    // write out chunks of U
    if(uipau0 >= 0)
      U[un + uipau0 * ((size_t)LDU)] = s_U[threadIdx.y + 0][threadIdx.x];
    if(uipau1 >= 0)
      U[un + uipau1 * ((size_t)LDU)] = s_U[threadIdx.y + 8][threadIdx.x];
    if(uipau2 >= 0)
      U[un + uipau2 * ((size_t)LDU)] = s_U[threadIdx.y + 16][threadIdx.x];
    if(uipau3 >= 0)
      U[un + uipau3 * ((size_t)LDU)] = s_U[threadIdx.y + 24][threadIdx.x];
  }
}

void HPL_dlaswp01T(const int  M,
                   const int  N,
                   double*    A,
                   const int  LDA,
                   double*    U,
                   const int  LDU,
                   const int* LINDXA,
                   const int* LINDXAU) {
  /*
   * Purpose
   * =======
   *
   * HPL_dlaswp01T copies  scattered rows  of  A  into an array U.  The
   * row offsets in  A  of the source rows  are specified by LINDXA.  The
   * destination of those rows are specified by  LINDXAU.  A
   * positive value of LINDXAU indicates that the array  destination is U,
   * and A otherwise. Rows of A are stored as columns in U.
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

  dim3 grid_size((M + TILE_DIM - 1) / TILE_DIM, (N + TILE_DIM - 1) / TILE_DIM);
  dim3 block_size(TILE_DIM, BLOCK_ROWS);
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
                     LINDXA,
                     LINDXAU);

  /*
   * End of HPL_dlaswp01T
   */
}
