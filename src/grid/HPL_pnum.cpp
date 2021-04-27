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

int HPL_pnum(const HPL_T_grid* GRID, const int MYROW, const int MYCOL) {
  /*
   * Purpose
   * =======
   *
   * HPL_pnum determines  the  rank  of a  process  as a function  of  its
   * coordinates in the grid.
   *
   * Arguments
   * =========
   *
   * GRID    (local input)                 const HPL_T_grid *
   *         On entry,  GRID  points  to the data structure containing the
   *         process grid information.
   *
   * MYROW   (local input)                 const int
   *         On entry,  MYROW  specifies the row coordinate of the process
   *         whose rank is to be determined. MYROW must be greater than or
   *         equal to zero and less than NPROW.
   *
   * MYCOL   (local input)                 const int
   *         On entry,  MYCOL  specifies  the  column  coordinate  of  the
   *         process whose rank is to be determined. MYCOL must be greater
   *         than or equal to zero and less than NPCOL.
   *
   * ---------------------------------------------------------------------
   */

  if(GRID->order == HPL_ROW_MAJOR)
    return (MYROW * GRID->npcol + MYCOL);
  else
    return (MYCOL * GRID->nprow + MYROW);
}
