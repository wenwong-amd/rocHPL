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

void HPL_ladd(int* J, int* K, int* I) {
  /*
   * Purpose
   * =======
   *
   * HPL_ladd adds  without carry two long positive integers  K and J and
   * puts the result into I. The long integers  I, J, K are encoded on 64
   * bits using an array of 2 integers.  The 32-lower bits  are stored in
   * the  first  entry  of each array,  the 32-higher bits  in the second
   * entry.
   *
   * Arguments
   * =========
   *
   * J       (local input)                 int *
   *         On entry, J is an integer array of dimension 2 containing the
   *         encoded long integer J.
   *
   * K       (local input)                 int *
   *         On entry, K is an integer array of dimension 2 containing the
   *         encoded long integer K.
   *
   * I       (local output)                int *
   *         On entry, I is an integer array of dimension 2. On exit, this
   *         array contains the encoded long integer result.
   *
   * ---------------------------------------------------------------------
   */

  unsigned int itmp0, itmp1;
  unsigned int ktmp0 = K[0] & 65535, ktmp1 = (unsigned)K[0] >> 16;
  unsigned int ktmp2 = K[1] & 65535, ktmp3 = (unsigned)K[1] >> 16;
  unsigned int jtmp0 = J[0] & 65535, jtmp1 = (unsigned)J[0] >> 16;
  unsigned int jtmp2 = J[1] & 65535, jtmp3 = (unsigned)J[1] >> 16;

  /*
   *    K[1] K[0] K  I[0]  = (K[0]+J[0]) % 2^32
   *    XXXX XXXX    carry = (K[0]+J[0]) / 2^32
   *
   * +  J[1] J[0] J  I[1] = K[1] + J[1] + carry
   *    XXXX XXXX    I[1] = I[1] % 2^32
   *    -------------
   *    I[1] I[0]
   *    0XXX XXXX I
   */
  itmp0 = ktmp0 + jtmp0;
  itmp1 = itmp0 >> 16;
  I[0]  = itmp0 - (itmp1 << 16);
  itmp1 += ktmp1 + jtmp1;
  I[0] |= (itmp1 & 65535) << 16;
  itmp0 = (itmp1 >> 16) + ktmp2 + jtmp2;
  I[1]  = itmp0 - ((itmp0 >> 16) << 16);
  itmp1 = (itmp0 >> 16) + ktmp3 + jtmp3;
  I[1] |= (itmp1 & 65535) << 16;
}
