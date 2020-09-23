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
/*
 * Define default value for unrolling factor
 */
#ifndef HPL_LASWP01N_DEPTH
#define HPL_LASWP01N_DEPTH 32
#define HPL_LASWP01N_LOG2_DEPTH 5
#endif

#define BLOCK_SIZE 512

__global__ void dlaswp01N(const int N,
                          const int M,
                          const int JB,
                          double* __restrict__ A,
                          const int LDA,
                          double* __restrict__ U,
                          const int LDU,
                          const int* __restrict__ LINDXA,
                          const int* __restrict__ LINDXAU) {

  __shared__ double s_An_init[2048];
  __shared__ double s_Un_ipiv[2048];

  const int m = threadIdx.x;
  const int n = blockIdx.x;

  // read in panel column of A
  for(int i = m; i < JB; i += blockDim.x)
    s_An_init[i] = A[i + n * ((size_t)LDA)];

  __syncthreads();

  // swap into U panel
  for(int i = m; i < M; i += blockDim.x) {
    const int ipa  = LINDXA[i];
    const int ipau = LINDXAU[i];

    if(ipau >= 0) {  // swap into U
      if(ipa < JB) { // take value from panel
        s_Un_ipiv[ipau] = s_An_init[ipa];
      } else { // take value from trailing A column
        s_Un_ipiv[ipau] = A[ipa + n * ((size_t)LDA)];
      }
    }
  }
  __syncthreads();

  // swap into trailing A column
  for(int i = m; i < M; i += blockDim.x) {
    const int ipa  = LINDXA[i];
    const int ipau = LINDXAU[i];

    if(ipau < 0) { // swap into A
      A[-ipau + n * ((size_t)LDA)] = s_An_init[ipa];
    }
  }

  // write out local panel column of U
  for(int i = m; i < JB; i += blockDim.x)
    U[i + n * ((size_t)LDU)] = s_Un_ipiv[i];
}

void HPL_dlaswp01N(const int  M,
                   const int  N,
                   const int  JB,
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
   * HPL_dlaswp01N copies  scattered rows  of  A  into itself  and into an
   * array  U.  The row offsets in  A  of the source rows are specified by
   * LINDXA.  The  destination of those rows are specified by  LINDXAU.  A
   * positive value of  LINDXAU indicates that the array destination is U,
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
   * U       (local input/output)          double *
   *         On entry, U points to an array of dimension (LDU,N). The rows
   *         of A specified by LINDXA are be copied within this array U at
   *         the positions indicated by positive values of LINDXAU.
   *
   * LDU     (local input)                 const int
   *         On entry, LDU specifies the leading dimension of the array U.
   *         LDU must be at least MAX(1,M).
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

  double *  a0, *a1;
  const int incA = (int)((unsigned int)(LDA) << HPL_LASWP01N_LOG2_DEPTH),
            incU = (int)((unsigned int)(LDU) << HPL_LASWP01N_LOG2_DEPTH);
  int lda1, nu, nr;
  int i, j;

  if((M <= 0) || (N <= 0)) return;

  hipStream_t stream;
  rocblas_get_stream(handle, &stream);

  int grid_size = N;
  hipLaunchKernelGGL((dlaswp01N),
                     dim3(grid_size),
                     dim3(BLOCK_SIZE),
                     0,
                     stream,
                     N,
                     M,
                     JB,
                     A,
                     LDA,
                     U,
                     LDU,
                     LINDXA,
                     LINDXAU);

// original
#if 0
   nr = N - ( nu = (int)( ( (unsigned int)(N) >> HPL_LASWP01N_LOG2_DEPTH ) <<
                            HPL_LASWP01N_LOG2_DEPTH ) );

   for( j = 0; j < nu; j += HPL_LASWP01N_DEPTH, A += incA, U += incU )
   {
      for( i = 0; i < M; i++ )
      {
         a0 = A + (size_t)(LINDXA[i]);
         if( LINDXAU[i] >= 0 ) { a1 = U + (size_t)(LINDXAU[i]); lda1 = LDU; }
         else                  { a1 = A - (size_t)(LINDXAU[i]); lda1 = LDA; }

         *a1 = *a0; a1 += lda1; a0 += LDA;
#if(HPL_LASWP01N_DEPTH > 1)
         *a1 = *a0; a1 += lda1; a0 += LDA;
#endif
#if(HPL_LASWP01N_DEPTH > 2)
         *a1 = *a0; a1 += lda1; a0 += LDA; *a1 = *a0; a1 += lda1; a0 += LDA;
#endif
#if(HPL_LASWP01N_DEPTH > 4)
         *a1 = *a0; a1 += lda1; a0 += LDA; *a1 = *a0; a1 += lda1; a0 += LDA;
         *a1 = *a0; a1 += lda1; a0 += LDA; *a1 = *a0; a1 += lda1; a0 += LDA;
#endif
#if(HPL_LASWP01N_DEPTH > 8)
         *a1 = *a0; a1 += lda1; a0 += LDA; *a1 = *a0; a1 += lda1; a0 += LDA;
         *a1 = *a0; a1 += lda1; a0 += LDA; *a1 = *a0; a1 += lda1; a0 += LDA;
         *a1 = *a0; a1 += lda1; a0 += LDA; *a1 = *a0; a1 += lda1; a0 += LDA;
         *a1 = *a0; a1 += lda1; a0 += LDA; *a1 = *a0; a1 += lda1; a0 += LDA;
#endif
#if(HPL_LASWP01N_DEPTH > 16)
         *a1 = *a0; a1 += lda1; a0 += LDA; *a1 = *a0; a1 += lda1; a0 += LDA;
         *a1 = *a0; a1 += lda1; a0 += LDA; *a1 = *a0; a1 += lda1; a0 += LDA;
         *a1 = *a0; a1 += lda1; a0 += LDA; *a1 = *a0; a1 += lda1; a0 += LDA;
         *a1 = *a0; a1 += lda1; a0 += LDA; *a1 = *a0; a1 += lda1; a0 += LDA;
         *a1 = *a0; a1 += lda1; a0 += LDA; *a1 = *a0; a1 += lda1; a0 += LDA;
         *a1 = *a0; a1 += lda1; a0 += LDA; *a1 = *a0; a1 += lda1; a0 += LDA;
         *a1 = *a0; a1 += lda1; a0 += LDA; *a1 = *a0; a1 += lda1; a0 += LDA;
         *a1 = *a0; a1 += lda1; a0 += LDA; *a1 = *a0; a1 += lda1; a0 += LDA;
#endif
      }
   }

   if( nr )
   {
      for( i = 0; i < M; i++ )
      {
         a0 = A + (size_t)(LINDXA[i]);
         if( LINDXAU[i] >= 0 ) { a1 = U + (size_t)(LINDXAU[i]); lda1 = LDU; }
         else                  { a1 = A - (size_t)(LINDXAU[i]); lda1 = LDA; }
         for( j = 0; j < nr; j++, a1 += lda1, a0 += LDA ) { *a1 = *a0; }
      }
   }
#endif
}
