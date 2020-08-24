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

int HPL_pdpanel_free(HPL_T_panel* PANEL) {
  /*
   * Purpose
   * =======
   *
   * HPL_pdpanel_free deallocates  the panel resources  and  stores the error
   * code returned by the panel factorization.
   *
   * Arguments
   * =========
   *
   * PANEL   (local input/output)          HPL_T_panel *
   *         On entry,  PANEL  points  to  the  panel data  structure from
   *         which the resources should be deallocated.
   *
   * ---------------------------------------------------------------------
   */

  if(PANEL->pmat->info == 0) PANEL->pmat->info = *(PANEL->DINFO);

  if(PANEL->free_work_now == 1) {
    if(PANEL->A) hipHostFree(PANEL->A);

    if(PANEL->WORK) hipHostFree(PANEL->WORK);

    if(PANEL->dWORK) hipFree(PANEL->dWORK);

    PANEL->max_work_size = 0;

    if(PANEL->IWORK) hipHostFree(PANEL->IWORK);

    PANEL->max_iwork_size = 0;
  }

  return (MPI_SUCCESS);
}
