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

void HPL_jumpit(int* MULT, int* IADD, int* IRANN, int* IRANM) {
  /*
   * Purpose
   * =======
   *
   * HPL_jumpit jumps in the random sequence from the number  X(n) encoded
   * in IRANN to the number  X(m)  encoded in  IRANM using the constants A
   * and C encoded in MULT and IADD: X(m) = A * X(n) + C.  The constants A
   * and C obviously depend on m and n,  see  the function  HPL_xjumpm  in
   * order to initialize them.
   *
   * Arguments
   * =========
   *
   * MULT    (local input)                 int *
   *         On entry, MULT is an array of dimension 2, that contains the
   *         16-lower and 15-higher bits of the constant A.
   *
   * IADD    (local input)                 int *
   *         On entry, IADD is an array of dimension 2, that contains the
   *         16-lower and 15-higher bits of the constant C.
   *
   * IRANN   (local input)                 int *
   *         On entry,  IRANN  is an array of dimension 2,  that contains
   *         the 16-lower and 15-higher bits of the encoding of X(n).
   *
   * IRANM   (local output)                int *
   *         On entry,  IRANM  is an array of dimension 2.  On exit, this
   *         array contains respectively the 16-lower and  15-higher bits
   *         of the encoding of X(m).
   *
   * ---------------------------------------------------------------------
   */

  int j[2];

  HPL_lmul(IRANN, MULT, j); /* j     = IRANN * MULT;  */
  HPL_ladd(j, IADD, IRANM); /* IRANM = j     + IADD;  */
  HPL_setran(0, IRANM);     /* irand = IRANM          */
}
