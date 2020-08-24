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

#include "hpl.h"
/*
 * Define default value for unrolling factor
 */
#ifndef HPL_LASWP05N_DEPTH
#define HPL_LASWP05N_DEPTH 32
#define HPL_LASWP05N_LOG2_DEPTH 5
#endif

void HPL_dlaswp05N(const int     M,
                   const int     N,
                   double*       A,
                   const int     LDA,
                   const double* U,
                   const int     LDU,
                   const int*    LINDXA,
                   const int*    LINDXAU) {
  /*
   * Purpose
   * =======
   *
   * HPL_dlaswp05N copies rows of  U of global offset LINDXAU into rows of
   * A at positions indicated by LINDXA.
   *
   * Arguments
   * =========
   *
   * M       (local input)                 const int
   *         On entry, M  specifies the number of rows of U that should be
   *         copied into A. M must be at least zero.
   *
   * N       (local input)                 const int
   *         On entry, N specifies the length of the rows of U that should
   *         be copied into A. N must be at least zero.
   *
   * A       (local output)                double *
   *         On entry, A points to an array of dimension (LDA,N). On exit,
   *         the  rows of this array specified by  LINDXA  are replaced by
   *         rows of U indicated by LINDXAU.
   *
   * LDA     (local input)                 const int
   *         On entry, LDA specifies the leading dimension of the array A.
   *         LDA must be at least MAX(1,M).
   *
   * U       (local input/output)          const double *
   *         On entry,  U  points to an array of dimension  (LDU,N).  This
   *         array contains the rows that are to be copied into A.
   *
   * LDU     (local input)                 const int
   *         On entry, LDU specifies the leading dimension of the array U.
   *         LDU must be at least MAX(1,M).
   *
   * LINDXA  (local input)                 const int *
   *         On entry, LINDXA is an array of dimension M that contains the
   *         local row indexes of A that should be copied from U.
   *
   * LINDXAU (local input)                 const int *
   *         On entry, LINDXAU  is an array of dimension  M that  contains
   *         the local row indexes of U that should be copied in A.
   *
   * ---------------------------------------------------------------------
   */

  const double *U0 = U, *u0;
  double*       a0;
  const int     incA = (int)((unsigned int)(LDA) << HPL_LASWP05N_LOG2_DEPTH),
            incU     = (int)((unsigned int)(LDU) << HPL_LASWP05N_LOG2_DEPTH);
  int nr, nu;
  int i, j;

  if((M <= 0) || (N <= 0)) return;

  nr = N - (nu = (int)(((unsigned int)(N) >> HPL_LASWP05N_LOG2_DEPTH)
                       << HPL_LASWP05N_LOG2_DEPTH));

  for(j = 0; j < nu; j += HPL_LASWP05N_DEPTH, A += incA, U0 += incU) {
    for(i = 0; i < M; i++) {
      a0 = A + (size_t)(LINDXA[i]);
      u0 = U0 + (size_t)(LINDXAU[i]);

      *a0 = *u0;
      a0 += LDA;
      u0 += LDU;
#if(HPL_LASWP05N_DEPTH > 1)
      *a0 = *u0;
      a0 += LDA;
      u0 += LDU;
#endif
#if(HPL_LASWP05N_DEPTH > 2)
      *a0 = *u0;
      a0 += LDA;
      u0 += LDU;
      *a0 = *u0;
      a0 += LDA;
      u0 += LDU;
#endif
#if(HPL_LASWP05N_DEPTH > 4)
      *a0 = *u0;
      a0 += LDA;
      u0 += LDU;
      *a0 = *u0;
      a0 += LDA;
      u0 += LDU;
      *a0 = *u0;
      a0 += LDA;
      u0 += LDU;
      *a0 = *u0;
      a0 += LDA;
      u0 += LDU;
#endif
#if(HPL_LASWP05N_DEPTH > 8)
      *a0 = *u0;
      a0 += LDA;
      u0 += LDU;
      *a0 = *u0;
      a0 += LDA;
      u0 += LDU;
      *a0 = *u0;
      a0 += LDA;
      u0 += LDU;
      *a0 = *u0;
      a0 += LDA;
      u0 += LDU;
      *a0 = *u0;
      a0 += LDA;
      u0 += LDU;
      *a0 = *u0;
      a0 += LDA;
      u0 += LDU;
      *a0 = *u0;
      a0 += LDA;
      u0 += LDU;
      *a0 = *u0;
      a0 += LDA;
      u0 += LDU;
#endif
#if(HPL_LASWP05N_DEPTH > 16)
      *a0 = *u0;
      a0 += LDA;
      u0 += LDU;
      *a0 = *u0;
      a0 += LDA;
      u0 += LDU;
      *a0 = *u0;
      a0 += LDA;
      u0 += LDU;
      *a0 = *u0;
      a0 += LDA;
      u0 += LDU;
      *a0 = *u0;
      a0 += LDA;
      u0 += LDU;
      *a0 = *u0;
      a0 += LDA;
      u0 += LDU;
      *a0 = *u0;
      a0 += LDA;
      u0 += LDU;
      *a0 = *u0;
      a0 += LDA;
      u0 += LDU;
      *a0 = *u0;
      a0 += LDA;
      u0 += LDU;
      *a0 = *u0;
      a0 += LDA;
      u0 += LDU;
      *a0 = *u0;
      a0 += LDA;
      u0 += LDU;
      *a0 = *u0;
      a0 += LDA;
      u0 += LDU;
      *a0 = *u0;
      a0 += LDA;
      u0 += LDU;
      *a0 = *u0;
      a0 += LDA;
      u0 += LDU;
      *a0 = *u0;
      a0 += LDA;
      u0 += LDU;
      *a0 = *u0;
      a0 += LDA;
      u0 += LDU;
#endif
    }
  }

  if(nr) {
    for(i = 0; i < M; i++) {
      a0 = A + (size_t)(LINDXA[i]);
      u0 = U0 + (size_t)(LINDXAU[i]);
      for(j = 0; j < nr; j++, a0 += LDA, u0 += LDU) { *a0 = *u0; }
    }
  }
}
