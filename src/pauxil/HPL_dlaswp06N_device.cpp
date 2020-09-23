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
#ifndef HPL_LASWP06N_DEPTH
#define HPL_LASWP06N_DEPTH 32
#define HPL_LASWP06N_LOG2_DEPTH 5
#endif

#define BLOCK_SIZE 512

__global__ void dlaswp06N(const int N,
                          const int M,
                          double* __restrict__ A,
                          const int LDA,
                          double* __restrict__ U,
                          const int LDU,
                          const int* __restrict__ LINDXA) {

  __shared__ double s_Un[2048];

  const int m = threadIdx.x;
  const int n = blockIdx.x;

  // read in block column
  for(int i = m; i < M; i += blockDim.x) s_Un[i] = U[i + n * ((size_t)LDU)];

  __syncthreads();

  // local block
  for(int i = m; i < M; i += blockDim.x) {
    const int ip = LINDXA[i];

    const double a0 = A[ip + n * ((size_t)LDA)];
    const double u0 = s_Un[i];

    // swap
    A[ip + n * ((size_t)LDA)] = u0;
    s_Un[i]                   = a0;
  }

  __syncthreads();

  // write out local block
  for(int i = m; i < M; i += blockDim.x) U[i + n * ((size_t)LDU)] = s_Un[i];
}

void HPL_dlaswp06N(const int  M,
                   const int  N,
                   double*    A,
                   const int  LDA,
                   double*    U,
                   const int  LDU,
                   const int* LINDXA) {
  /*
   * Purpose
   * =======
   *
   * HPL_dlaswp06N swaps rows of  U  with rows of A at positions
   * indicated by LINDXA.
   *
   * Arguments
   * =========
   *
   * M       (local input)                 const int
   *         On entry, M  specifies the number of rows of A that should be
   *         swapped with rows of U. M must be at least zero.
   *
   * N       (local input)                 const int
   *         On entry, N specifies the length of the rows of A that should
   *         be swapped with rows of U. N must be at least zero.
   *
   * A       (local output)                double *
   *         On entry, A points to an array of dimension (LDA,N). On exit,
   *         the  rows of this array specified by  LINDXA  are replaced by
   *         rows or columns of U.
   *
   * LDA     (local input)                 const int
   *         On entry, LDA specifies the leading dimension of the array A.
   *         LDA must be at least MAX(1,M).
   *
   * U       (local input/output)          double *
   *         On entry,  U  points  to an array of dimension (LDU,N).  This
   *         array contains the rows of U that are to be swapped with rows
   *         of A.
   *
   * LDU     (local input)                 const int
   *         On entry, LDU specifies the leading dimension of the array U.
   *         LDU must be at least MAX(1,M).
   *
   * LINDXA  (local input)                 const int *
   *         On entry, LINDXA is an array of dimension M that contains the
   *         local row indexes of A that should be swapped with U.
   *
   * ---------------------------------------------------------------------
   */

  double    r;
  double *  U0   = U, *a0, *u0;
  const int incA = (int)((unsigned int)(LDA) << HPL_LASWP06N_LOG2_DEPTH),
            incU = (int)((unsigned int)(LDU) << HPL_LASWP06N_LOG2_DEPTH);
  int nr, nu;
  int i, j;

  if((M <= 0) || (N <= 0)) return;

  hipStream_t stream;
  rocblas_get_stream(handle, &stream);

  int grid_size = N;
  hipLaunchKernelGGL((dlaswp06N),
                     dim3(grid_size),
                     dim3(BLOCK_SIZE),
                     0,
                     stream,
                     N,
                     M,
                     A,
                     LDA,
                     U,
                     LDU,
                     LINDXA);

// original
#if 0
   nr = N - ( nu = (int)( ( (unsigned int)(N) >> HPL_LASWP06N_LOG2_DEPTH ) <<
                            HPL_LASWP06N_LOG2_DEPTH ) );

   for( j = 0; j < nu; j += HPL_LASWP06N_DEPTH, A += incA, U0 += incU )
   {
      for( i = 0; i < M; i++ )
      {
         a0 = A + (size_t)(LINDXA[i]); u0 = U0 + (size_t)(i);

         r = *a0; *a0 = *u0; *u0 = r; a0 += LDA; u0 += LDU;
#if(HPL_LASWP06N_DEPTH > 1)
         r = *a0; *a0 = *u0; *u0 = r; a0 += LDA; u0 += LDU;
#endif
#if(HPL_LASWP06N_DEPTH > 2)
         r = *a0; *a0 = *u0; *u0 = r; a0 += LDA; u0 += LDU;
         r = *a0; *a0 = *u0; *u0 = r; a0 += LDA; u0 += LDU;
#endif
#if(HPL_LASWP06N_DEPTH > 4)
         r = *a0; *a0 = *u0; *u0 = r; a0 += LDA; u0 += LDU;
         r = *a0; *a0 = *u0; *u0 = r; a0 += LDA; u0 += LDU;
         r = *a0; *a0 = *u0; *u0 = r; a0 += LDA; u0 += LDU;
         r = *a0; *a0 = *u0; *u0 = r; a0 += LDA; u0 += LDU;
#endif
#if(HPL_LASWP06N_DEPTH > 8)
         r = *a0; *a0 = *u0; *u0 = r; a0 += LDA; u0 += LDU;
         r = *a0; *a0 = *u0; *u0 = r; a0 += LDA; u0 += LDU;
         r = *a0; *a0 = *u0; *u0 = r; a0 += LDA; u0 += LDU;
         r = *a0; *a0 = *u0; *u0 = r; a0 += LDA; u0 += LDU;
         r = *a0; *a0 = *u0; *u0 = r; a0 += LDA; u0 += LDU;
         r = *a0; *a0 = *u0; *u0 = r; a0 += LDA; u0 += LDU;
         r = *a0; *a0 = *u0; *u0 = r; a0 += LDA; u0 += LDU;
         r = *a0; *a0 = *u0; *u0 = r; a0 += LDA; u0 += LDU;
#endif
#if(HPL_LASWP06N_DEPTH > 16)
         r = *a0; *a0 = *u0; *u0 = r; a0 += LDA; u0 += LDU;
         r = *a0; *a0 = *u0; *u0 = r; a0 += LDA; u0 += LDU;
         r = *a0; *a0 = *u0; *u0 = r; a0 += LDA; u0 += LDU;
         r = *a0; *a0 = *u0; *u0 = r; a0 += LDA; u0 += LDU;
         r = *a0; *a0 = *u0; *u0 = r; a0 += LDA; u0 += LDU;
         r = *a0; *a0 = *u0; *u0 = r; a0 += LDA; u0 += LDU;
         r = *a0; *a0 = *u0; *u0 = r; a0 += LDA; u0 += LDU;
         r = *a0; *a0 = *u0; *u0 = r; a0 += LDA; u0 += LDU;
         r = *a0; *a0 = *u0; *u0 = r; a0 += LDA; u0 += LDU;
         r = *a0; *a0 = *u0; *u0 = r; a0 += LDA; u0 += LDU;
         r = *a0; *a0 = *u0; *u0 = r; a0 += LDA; u0 += LDU;
         r = *a0; *a0 = *u0; *u0 = r; a0 += LDA; u0 += LDU;
         r = *a0; *a0 = *u0; *u0 = r; a0 += LDA; u0 += LDU;
         r = *a0; *a0 = *u0; *u0 = r; a0 += LDA; u0 += LDU;
         r = *a0; *a0 = *u0; *u0 = r; a0 += LDA; u0 += LDU;
         r = *a0; *a0 = *u0; *u0 = r; a0 += LDA; u0 += LDU;
#endif
      }
   }

   if( nr )
   {
      for( i = 0; i < M; i++ )
      {
         a0 = A + (size_t)(LINDXA[i]); u0 = U0 + (size_t)(i);
         for( j = 0; j < nr; j++, a0 += LDA, u0 += LDU )
         { r = *a0; *a0 = *u0; *u0 = r; }
      }
   }
#endif
}
