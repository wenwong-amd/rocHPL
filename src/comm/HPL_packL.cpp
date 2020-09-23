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

int HPL_packL(HPL_T_panel* PANEL,
              const int    INDEX,
              const int    LEN,
              const int    IBUF) {
  /*
   * Purpose
   * =======
   *
   * HPL_packL forms  the MPI data type for the panel to be broadcast.
   * Successful  completion  is  indicated  by  the  returned  error  code
   * MPI_SUCCESS.
   *
   * Arguments
   * =========
   *
   * PANEL   (input/output)                HPL_T_panel *
   *         On entry,  PANEL  points to the  current panel data structure
   *         being broadcast.
   *
   * INDEX   (input)                       const int
   *         On entry,  INDEX  points  to  the  first entry of the  packed
   *         buffer being broadcast.
   *
   * LEN     (input)                       const int
   *         On entry, LEN is the length of the packed buffer.
   *
   * IBUF    (input)                       const int
   *         On entry, IBUF  specifies the panel buffer/count/type entries
   *         that should be initialized.
   *
   * ---------------------------------------------------------------------
   */
  return (MPI_SUCCESS);
}
