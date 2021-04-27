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

void HPL_lmul(int* K, int* J, int* I) {
  /*
   * Purpose
   * =======
   *
   * HPL_lmul multiplies  without carry two long positive integers K and J
   * and puts the result into I. The long integers  I, J, K are encoded on
   * 64 bits using an array of 2 integers. The 32-lower bits are stored in
   * the first entry of each array, the 32-higher bits in the second entry
   * of each array. For efficiency purposes, the  intrisic modulo function
   * is inlined.
   *
   * Arguments
   * =========
   *
   * K       (local input)                 int *
   *         On entry, K is an integer array of dimension 2 containing the
   *         encoded long integer K.
   *
   * J       (local input)                 int *
   *         On entry, J is an integer array of dimension 2 containing the
   *         encoded long integer J.
   *
   * I       (local output)                int *
   *         On entry, I is an integer array of dimension 2. On exit, this
   *         array contains the encoded long integer result.
   *
   * ---------------------------------------------------------------------
   */

  int          r, c;
  unsigned int kk[4], jj[4], res[5];

  /*
   * Addition is done with 16 bits at a time. Multiplying two 16-bit
   * integers yields a 32-bit result. The lower 16-bits of the result
   * are kept in I, and the higher 16-bits are carried over to the
   * next multiplication.
   */
  for(c = 0; c < 2; ++c) {
    kk[2 * c]     = K[c] & 65535;
    kk[2 * c + 1] = ((unsigned)K[c] >> 16) & 65535;
    jj[2 * c]     = J[c] & 65535;
    jj[2 * c + 1] = ((unsigned)J[c] >> 16) & 65535;
  }

  res[0] = 0;
  for(c = 0; c < 4; ++c) {
    res[c + 1] = (res[c] >> 16) & 65535;
    res[c] &= 65535;
    for(r = 0; r < c + 1; ++r) {
      res[c] = kk[r] * jj[c - r] + (res[c] & 65535);
      res[c + 1] += (res[c] >> 16) & 65535;
    }
  }

  for(c = 0; c < 2; ++c)
    I[c] = (int)(((res[2 * c + 1] & 65535) << 16) | (res[2 * c] & 65535));
}
