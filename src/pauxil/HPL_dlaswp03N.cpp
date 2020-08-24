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
#ifndef HPL_LASWP03N_DEPTH
#define HPL_LASWP03N_DEPTH 32
#define HPL_LASWP03N_LOG2_DEPTH 5
#endif

void HPL_dlaswp03N(const int     M,
                   const int     N,
                   double*       U,
                   const int     LDU,
                   const double* W0,
                   const double* W,
                   const int     LDW) {
  /*
   * Purpose
   * =======
   *
   * HPL_dlaswp03N copies columns of  W  into  rows  of an  array  U.  The
   * destination in U of these columns contained in W is stored within W0.
   *
   * Arguments
   * =========
   *
   * M       (local input)                 const int
   *         On entry, M  specifies  the  number  of columns of  W  stored
   *         contiguously that should be copied into U. M must be at least
   *         zero.
   *
   * N       (local input)                 const int
   *         On entry,  N  specifies  the  length of columns of  W  stored
   *         contiguously that should be copied into U. N must be at least
   *         zero.
   *
   * U       (local input/output)          double *
   *         On entry, U points to an array of dimension (LDU,N).  Columns
   *         of W are copied as rows within this array U at  the positions
   *         specified in W0.
   *
   * LDU     (local input)                 const int
   *         On entry, LDU specifies the leading dimension of the array U.
   *         LDU must be at least MAX(1,M).
   *
   * W0      (local input)                 const double *
   *         On entry,  W0  is an array of size (M-1)*LDW+1, that contains
   *         the destination offset  in U where the columns of W should be
   *         copied.
   *
   * W       (local input)                 const double *
   *         On entry, W  is an array of size (LDW,M),  that contains data
   *         to be copied into U. For i in [0..M),  entries W(:,i)  should
   *         be copied into the row or column W0(i*LDW) of U.
   *
   * LDW     (local input)                 const int
   *         On entry, LDW specifies the leading dimension of the array W.
   *         LDW must be at least MAX(1,N+1).
   *
   * ---------------------------------------------------------------------
   */

  const double *w = W, *w0;
  double*       u0;
  const int     incU = (int)((unsigned int)(LDU) << HPL_LASWP03N_LOG2_DEPTH);
  int           nr, nu;
  int           i, j;

  if((M <= 0) || (N <= 0)) return;

  nr = N - (nu = (int)(((unsigned int)(N) >> HPL_LASWP03N_LOG2_DEPTH)
                       << HPL_LASWP03N_LOG2_DEPTH));

  for(j = 0; j < nu;
      j += HPL_LASWP03N_DEPTH, U += incU, w += HPL_LASWP03N_DEPTH) {
    for(i = 0; i < M; i++) {
      u0 = U + (size_t)(*(W0 + (size_t)(i) * (size_t)(LDW)));
      w0 = w + (size_t)(i) * (size_t)(LDW);

      *u0 = w0[0];
      u0 += LDU;
#if(HPL_LASWP03N_DEPTH > 1)
      *u0 = w0[1];
      u0 += LDU;
#endif
#if(HPL_LASWP03N_DEPTH > 2)
      *u0 = w0[2];
      u0 += LDU;
      *u0 = w0[3];
      u0 += LDU;
#endif
#if(HPL_LASWP03N_DEPTH > 4)
      *u0 = w0[4];
      u0 += LDU;
      *u0 = w0[5];
      u0 += LDU;
      *u0 = w0[6];
      u0 += LDU;
      *u0 = w0[7];
      u0 += LDU;
#endif
#if(HPL_LASWP03N_DEPTH > 8)
      *u0 = w0[8];
      u0 += LDU;
      *u0 = w0[9];
      u0 += LDU;
      *u0 = w0[10];
      u0 += LDU;
      *u0 = w0[11];
      u0 += LDU;
      *u0 = w0[12];
      u0 += LDU;
      *u0 = w0[13];
      u0 += LDU;
      *u0 = w0[14];
      u0 += LDU;
      *u0 = w0[15];
      u0 += LDU;
#endif
#if(HPL_LASWP03N_DEPTH > 16)
      *u0 = w0[16];
      u0 += LDU;
      *u0 = w0[17];
      u0 += LDU;
      *u0 = w0[18];
      u0 += LDU;
      *u0 = w0[19];
      u0 += LDU;
      *u0 = w0[20];
      u0 += LDU;
      *u0 = w0[21];
      u0 += LDU;
      *u0 = w0[22];
      u0 += LDU;
      *u0 = w0[23];
      u0 += LDU;
      *u0 = w0[24];
      u0 += LDU;
      *u0 = w0[25];
      u0 += LDU;
      *u0 = w0[26];
      u0 += LDU;
      *u0 = w0[27];
      u0 += LDU;
      *u0 = w0[28];
      u0 += LDU;
      *u0 = w0[29];
      u0 += LDU;
      *u0 = w0[30];
      u0 += LDU;
      *u0 = w0[31];
      u0 += LDU;
#endif
    }
  }

  if(nr) {
    for(i = 0; i < M; i++) {
      u0 = U + (size_t)(*(W0 + (size_t)(i) * (size_t)(LDW)));
      w0 = w + (size_t)(i) * (size_t)(LDW);
      for(j = 0; j < nr; j++, u0 += LDU) { *u0 = w0[j]; }
    }
  }
}
