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
/*
 * ---------------------------------------------------------------------
 * Static variables
 * ---------------------------------------------------------------------
 */
static int ias[2], ics[2], irand[2];

void HPL_setran(const int OPTION, int* IRAN) {
  /*
   * Purpose
   * =======
   *
   * HPL_setran initializes  the random generator with the encoding of the
   * first number X(0) in the sequence,  and the constants a and c used to
   * compute the next element in the sequence: X(n+1) = a*X(n) + c.  X(0),
   * a and c are stored in the static variables  irand, ias and ics.  When
   * OPTION is 0 (resp. 1 and 2),  irand  (resp. ia and ic)  is set to the
   * values of the input array IRAN.  When OPTION is 3, IRAN is set to the
   * current value of irand, and irand is then incremented.
   *
   * Arguments
   * =========
   *
   * OPTION  (local input)                 const int
   *         On entry, OPTION  is an integer that specifies the operations
   *         to be performed on the random generator as specified above.
   *
   * IRAN    (local input/output)          int *
   *         On entry,  IRAN is an array of dimension 2, that contains the
   *         16-lower and 15-higher bits of a random number.
   *
   * ---------------------------------------------------------------------
   */

  int j[2];

  if(OPTION == 3) { /* return current value */
    IRAN[0] = irand[0];
    IRAN[1] = irand[1];
    HPL_lmul(irand, ias, j); /* j     = irand * ias;   */
    HPL_ladd(j, ics, irand); /* irand = j     + ics;   */
  } else if(OPTION == 0) {
    irand[0] = IRAN[0];
    irand[1] = IRAN[1];
  } else if(OPTION == 1) {
    ias[0] = IRAN[0];
    ias[1] = IRAN[1];
  } else if(OPTION == 2) {
    ics[0] = IRAN[0];
    ics[1] = IRAN[1];
  }
}
