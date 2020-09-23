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

void HPL_copyL(HPL_T_panel* PANEL) {
  /*
   * Purpose
   * =======
   *
   * HPL_copyL copies  the  panel of columns, the L1 replicated submatrix,
   * the pivot array  and  the info scalar into a contiguous workspace for
   * later broadcast.
   *
   * The copy of this panel  into  a contiguous buffer  can be enforced by
   * specifying -DHPL_COPY_L in the architecture specific Makefile.
   *
   * Arguments
   * =========
   *
   * PANEL   (input/output)                HPL_T_panel *
   *         On entry,  PANEL  points to the  current panel data structure
   *         being broadcast.
   *
   * ---------------------------------------------------------------------
   */

  int jb, lda;

  if(PANEL->grid->mycol == PANEL->pcol) {
    jb  = PANEL->jb;
    lda = PANEL->lda;

    if(PANEL->grid->myrow == PANEL->prow) {
#if !defined(GPU_AWARE_MPI)
      HPL_dlacpy(PANEL->mp - jb,
                 jb,
                 Mptr(PANEL->A, jb, 0, lda),
                 lda,
                 PANEL->L2,
                 PANEL->ldl2);
#endif
    } else {
#if !defined(GPU_AWARE_MPI)
      HPL_dlacpy(PANEL->mp,
                 jb,
                 Mptr(PANEL->A, 0, 0, lda),
                 lda,
                 PANEL->L2,
                 PANEL->ldl2);
#endif
    }
  }
}
