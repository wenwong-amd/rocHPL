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

void HPL_dmatgen(const int M,
                 const int N,
                 double*   A,
                 const int LDA,
                 const int ISEED) {
  /*
   * Purpose
   * =======
   *
   * HPL_dmatgen generates (or regenerates) a random matrix A.
   *
   * The  pseudo-random  generator uses the linear congruential algorithm:
   * X(n+1) = (a * X(n) + c) mod m  as  described  in the  Art of Computer
   * Programming, Knuth 1973, Vol. 2.
   *
   * Arguments
   * =========
   *
   * M       (input)                       const int
   *         On entry,  M  specifies  the number  of rows of the matrix A.
   *         M must be at least zero.
   *
   * N       (input)                       const int
   *         On entry,  N specifies the number of columns of the matrix A.
   *         N must be at least zero.
   *
   * A       (output)                      double *
   *         On entry, A points to an array of dimension (LDA,N). On exit,
   *         this  array  contains   the   coefficients  of  the  randomly
   *         generated matrix.
   *
   * LDA     (input)                       const int
   *         On entry, LDA specifies the leading dimension of the array A.
   *         LDA must be at least max(1,M).
   *
   * ISEED   (input)                       const int
   *         On entry, ISEED  specifies  the  seed  number to generate the
   *         matrix A. ISEED must be at least zero.
   *
   * ---------------------------------------------------------------------
   */

  int iadd[2], ia1[2], ic1[2], iran1[2], jseed[2], mult[2];
  int i, incA = LDA - M, j;

  if((M <= 0) || (N <= 0)) return;
  /*
   * Initialize the random sequence
   */
  mult[0]  = HPL_MULT0;
  mult[1]  = HPL_MULT1;
  iadd[0]  = HPL_IADD0;
  iadd[1]  = HPL_IADD1;
  jseed[0] = ISEED;
  jseed[1] = 0;

  HPL_xjumpm(1, mult, iadd, jseed, iran1, ia1, ic1);
  HPL_setran(0, iran1);
  HPL_setran(1, ia1);
  HPL_setran(2, ic1);
  /*
   * Generate an M by N matrix
   */
  for(j = 0; j < N; A += incA, j++)
    for(i = 0; i < M; A++, i++) *A = HPL_rand();
}
