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
#ifndef HPL_LASWP05T_DEPTH
#define HPL_LASWP05T_DEPTH 32
#define HPL_LASWP05T_LOG2_DEPTH 5
#endif

void HPL_dlaswp05T(const int     M,
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
   * HPL_dlaswp05T copies columns of  U of global offset LINDXAU into rows
   * of A at positions indicated by LINDXA.
   *
   * Arguments
   * =========
   *
   * M       (local input)                 const int
   *         On entry,  M  specifies the number of columns of U that shouldbe
   * copied into A. M must be at least zero.
   *
   * N       (local input)                 const int
   *         On entry, N specifies the length of the columns of U that will
   *         be copied into rows of A. N must be at least zero.
   *
   * A       (local output)                double *
   *         On entry, A points to an array of dimension (LDA,N). On exit,
   *         the  rows of this array specified by  LINDXA  are replaced by
   *         columns of U indicated by LINDXAU.
   *
   * LDA     (local input)                 const int
   *         On entry, LDA specifies the leading dimension of the array A.
   *         LDA must be at least MAX(1,M).
   *
   * U       (local input/output)          const double *
   *         On entry,  U  points  to an array of dimension (LDU,*).  This
   *         array contains the columns that are to be copied into rows of
   *         A.
   *
   * LDU     (local input)                 const int
   *         On entry, LDU specifies the leading dimension of the array U.
   *         LDU must be at least MAX(1,N).
   *
   * LINDXA  (local input)                 const int *
   *         On entry, LINDXA is an array of dimension M that contains the
   *         local row indexes of A that should be copied from U.
   *
   * LINDXAU (local input)                 const int *
   *         On entry, LINDXAU  is an array of dimension  M that  contains
   *         the local column indexes of U that should be copied in A.
   *
   * ---------------------------------------------------------------------
   */

  const double *U0 = U, *u0;
  double*       a0;
  const int     incA = (int)((unsigned int)(LDA) << HPL_LASWP05T_LOG2_DEPTH),
            incU     = (1 << HPL_LASWP05T_LOG2_DEPTH);
  int nr, nu;
  int i, j;

  if((M <= 0) || (N <= 0)) return;

  nr = N - (nu = (int)(((unsigned int)(N) >> HPL_LASWP05T_LOG2_DEPTH)
                       << HPL_LASWP05T_LOG2_DEPTH));

  for(j = 0; j < nu; j += HPL_LASWP05T_DEPTH, A += incA, U0 += incU) {
    for(i = 0; i < M; i++) {
      a0 = A + (size_t)(LINDXA[i]);
      u0 = U0 + (size_t)(LINDXAU[i]) * (size_t)(LDU);

      *a0 = u0[0];
      a0 += LDA;
#if(HPL_LASWP05T_DEPTH > 1)
      *a0 = u0[1];
      a0 += LDA;
#endif
#if(HPL_LASWP05T_DEPTH > 2)
      *a0 = u0[2];
      a0 += LDA;
      *a0 = u0[3];
      a0 += LDA;
#endif
#if(HPL_LASWP05T_DEPTH > 4)
      *a0 = u0[4];
      a0 += LDA;
      *a0 = u0[5];
      a0 += LDA;
      *a0 = u0[6];
      a0 += LDA;
      *a0 = u0[7];
      a0 += LDA;
#endif
#if(HPL_LASWP05T_DEPTH > 8)
      *a0 = u0[8];
      a0 += LDA;
      *a0 = u0[9];
      a0 += LDA;
      *a0 = u0[10];
      a0 += LDA;
      *a0 = u0[11];
      a0 += LDA;
      *a0 = u0[12];
      a0 += LDA;
      *a0 = u0[13];
      a0 += LDA;
      *a0 = u0[14];
      a0 += LDA;
      *a0 = u0[15];
      a0 += LDA;
#endif
#if(HPL_LASWP05T_DEPTH > 16)
      *a0 = u0[16];
      a0 += LDA;
      *a0 = u0[17];
      a0 += LDA;
      *a0 = u0[18];
      a0 += LDA;
      *a0 = u0[19];
      a0 += LDA;
      *a0 = u0[20];
      a0 += LDA;
      *a0 = u0[21];
      a0 += LDA;
      *a0 = u0[22];
      a0 += LDA;
      *a0 = u0[23];
      a0 += LDA;
      *a0 = u0[24];
      a0 += LDA;
      *a0 = u0[25];
      a0 += LDA;
      *a0 = u0[26];
      a0 += LDA;
      *a0 = u0[27];
      a0 += LDA;
      *a0 = u0[28];
      a0 += LDA;
      *a0 = u0[29];
      a0 += LDA;
      *a0 = u0[30];
      a0 += LDA;
      *a0 = u0[31];
      a0 += LDA;
#endif
    }
  }

  if(nr > 0) {
    for(i = 0; i < M; i++) {
      a0 = A + (size_t)(LINDXA[i]);
      u0 = U0 + (size_t)(LINDXAU[i]) * (size_t)(LDU);
      for(j = 0; j < nr; j++, a0 += LDA) { *a0 = u0[j]; }
    }
  }
}
