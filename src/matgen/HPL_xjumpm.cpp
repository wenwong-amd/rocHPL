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

void HPL_xjumpm(const int JUMPM,
                int*      MULT,
                int*      IADD,
                int*      IRANN,
                int*      IRANM,
                int*      IAM,
                int*      ICM) {
  /*
   * Purpose
   * =======
   *
   * HPL_xjumpm computes  the constants  A and C  to jump JUMPM numbers in
   * the random sequence: X(n+JUMPM) = A*X(n)+C.  The constants encoded in
   * MULT and IADD  specify  how to jump from one entry in the sequence to
   * the next.
   *
   * Arguments
   * =========
   *
   * JUMPM   (local input)                 const int
   *         On entry,  JUMPM  specifies  the  number  of entries  in  the
   *         sequence to jump over. When JUMPM is less or equal than zero,
   *         A and C are not computed, IRANM is set to IRANN corresponding
   *         to a jump of size zero.
   *
   * MULT    (local input)                 int *
   *         On entry, MULT is an array of dimension 2,  that contains the
   *         16-lower  and 15-higher bits of the constant  a  to jump from
   *         X(n) to X(n+1) = a*X(n) + c in the random sequence.
   *
   * IADD    (local input)                 int *
   *         On entry, IADD is an array of dimension 2,  that contains the
   *         16-lower  and 15-higher bits of the constant  c  to jump from
   *         X(n) to X(n+1) = a*X(n) + c in the random sequence.
   *
   * IRANN   (local input)                 int *
   *         On entry, IRANN is an array of dimension 2. that contains the
   *         16-lower and 15-higher bits of the encoding of X(n).
   *
   * IRANM   (local output)                int *
   *         On entry,  IRANM  is an array of dimension 2.   On exit, this
   *         array  contains respectively  the 16-lower and 15-higher bits
   *         of the encoding of X(n+JUMPM).
   *
   * IAM     (local output)                int *
   *         On entry, IAM is an array of dimension 2. On exit, when JUMPM
   *         is  greater  than  zero,  this  array  contains  the  encoded
   *         constant  A  to jump from  X(n) to  X(n+JUMPM)  in the random
   *         sequence. IAM(0:1)  contains  respectively  the  16-lower and
   *         15-higher  bits  of this constant  A. When  JUMPM  is less or
   *         equal than zero, this array is not referenced.
   *
   * ICM     (local output)                int *
   *         On entry, ICM is an array of dimension 2. On exit, when JUMPM
   *         is  greater  than  zero,  this  array  contains  the  encoded
   *         constant  C  to jump from  X(n)  to  X(n+JUMPM) in the random
   *         sequence. ICM(0:1)  contains  respectively  the  16-lower and
   *         15-higher  bits  of this constant  C. When  JUMPM  is less or
   *         equal than zero, this array is not referenced.
   *
   * ---------------------------------------------------------------------
   */

  int j[2], k;

  if(JUMPM > 0) {
    IAM[0] = MULT[0];
    IAM[1] = MULT[1]; /* IAM   = MULT;          */
    ICM[0] = IADD[0];
    ICM[1] = IADD[1]; /* ICM   = IADD;          */
    for(k = 1; k <= JUMPM - 1; k++) {
      HPL_lmul(IAM, MULT, j); /* j     = IAM   * MULT;  */
      IAM[0] = j[0];
      IAM[1] = j[1];          /* IAM   = j;             */
      HPL_lmul(ICM, MULT, j); /* j     = ICM   * MULT;  */
      HPL_ladd(IADD, j, ICM); /* ICM   = IADD  + j;     */
    }
    HPL_lmul(IRANN, IAM, j); /* j     = IRANN * IAM;   */
    HPL_ladd(j, ICM, IRANM); /* IRANM = j     + ICM;   */
  } else {                   /* IRANM = IRANN          */
    IRANM[0] = IRANN[0];
    IRANM[1] = IRANN[1];
  }
}
