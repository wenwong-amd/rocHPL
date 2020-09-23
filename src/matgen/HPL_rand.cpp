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

double HPL_rand(void) {
  /*
   * Purpose
   * =======
   *
   * HPL_rand generates  the next number  in the  random  sequence.  This
   * function  ensures  that this number lies in the interval (-0.5, 0.5].
   *
   * The static array irand contains the information (2 integers) required
   * to generate the  next number  in the sequence  X(n).  This  number is
   * computed as X(n) = (2^32 * irand[1] + irand[0]) / d - 0.5,  where the
   * constant d is the largest 64 bit positive unsigned integer. The array
   * irand is then  updated  for the generation of the next number  X(n+1)
   * in  the  random   sequence  as   follows  X(n+1) = a * X(n) + c.  The
   * constants a and c should have been preliminarily stored in the arrays
   * ias and ics as 2 pairs of integers.  The initialization of  ias,  ics
   * and  irand  is performed by the function HPL_setran.
   *
   * ---------------------------------------------------------------------
   */

  int j[2];

  HPL_setran(3, j);
  /*
   * return number between -0.5 and 0.5
   */
  return (HPL_HALF - (((j[0] & 65535) + ((unsigned)j[0] >> 16) * HPL_POW16) /
                          HPL_DIVFAC * HPL_HALF +
                      (j[1] & 65535) + ((unsigned)j[1] >> 16) * HPL_POW16) /
                         HPL_DIVFAC * HPL_HALF);
}
