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

void HPL_max(const int N, const void* IN, void* INOUT, const HPL_T_TYPE DTYPE) {
  /*
   * Purpose
   * =======
   *
   * HPL_max combines (max) two buffers.
   *
   *
   * Arguments
   * =========
   *
   * N       (input)                       const int
   *         On entry, N  specifies  the  length  of  the  buffers  to  be
   *         combined. N must be at least zero.
   *
   * IN      (input)                       const void *
   *         On entry, IN points to the input-only buffer to be combined.
   *
   * INOUT   (input/output)                void *
   *         On entry, INOUT  points  to  the  input-output  buffer  to be
   *         combined.  On exit,  the  entries of this array contains  the
   *         combined results.
   *
   * DTYPE   (input)                       const HPL_T_TYPE
   *         On entry,  DTYPE  specifies the type of the buffers operands.
   *
   * ---------------------------------------------------------------------
   */

  int i;

  if(DTYPE == HPL_INT) {
    const int* a = (const int*)(IN);
    int*       b = (int*)(INOUT);
    for(i = 0; i < N; i++) b[i] = Mmax(a[i], b[i]);
  } else {
    const double* a = (const double*)(IN);
    double*       b = (double*)(INOUT);
    for(i = 0; i < N; i++) b[i] = Mmax(a[i], b[i]);
  }
}
