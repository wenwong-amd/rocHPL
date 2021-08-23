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

int HPL_bcast(HPL_T_panel* PANEL, int* IFLAG) {
  /*
   * Purpose
   * =======
   *
   * HPL_bcast broadcasts  the  current  panel.  Successful  completion is
   * indicated by IFLAG set to HPL_SUCCESS on return. IFLAG will be set to
   * HPL_FAILURE on failure and to HPL_KEEP_TESTING when the operation was
   * not completed, in which case this function should be called again.
   *
   * Arguments
   * =========
   *
   * PANEL   (input/output)                HPL_T_panel *
   *         On entry,  PANEL  points to the  current panel data structure
   *         being broadcast.
   *
   * IFLAG   (output)                      int *
   *         On exit,  IFLAG  indicates  whether  or not the broadcast has
   *         occured.
   *
   * ---------------------------------------------------------------------
   */

  MPI_Comm comm;
  int      ierr, ierr2, go, next, msgid, prev, rank, root, size;

  if(PANEL == NULL) {
    *IFLAG = HPL_SUCCESS;
    return (HPL_SUCCESS);
  }
  if((size = PANEL->grid->npcol) <= 1) {
    *IFLAG = HPL_SUCCESS;
    return (HPL_SUCCESS);
  }

  rank  = PANEL->grid->mycol;
  comm  = PANEL->grid->row_comm;
  root  = PANEL->pcol;
  msgid = PANEL->msgid;

  /*
   * Force the copy of the panel into a contiguous buffer
   */
  HPL_copyL(PANEL);

  roctxRangePush("MPI_Bcast");
  /*
   * Single Bcast call
   */
#if defined(GPU_AWARE_MPI)
  ierr = MPI_Bcast(PANEL->dL2, PANEL->len, MPI_DOUBLE, root, comm);
#else
  ierr = MPI_Bcast(PANEL->L2, PANEL->len, MPI_DOUBLE, root, comm);
#endif

  roctxRangePop();
  /*
   * If an error occured in an MPI call, return HPL_FAILURE.
   */
  *IFLAG = (ierr == MPI_SUCCESS ? HPL_SUCCESS : HPL_FAILURE);

  return (*IFLAG);
}
