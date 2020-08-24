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

  int       ierr;
  HPL_T_TOP top;

  if(PANEL == NULL) {
    *IFLAG = HPL_SUCCESS;
    return (HPL_SUCCESS);
  }
  if(PANEL->grid->npcol <= 1) {
    *IFLAG = HPL_SUCCESS;
    return (HPL_SUCCESS);
  }
  /*
   * Retrieve the selected virtual broadcast topology
   */
  top = PANEL->algo->btopo;

  switch(top) {
    case HPL_1RING_M: ierr = HPL_bcast_1rinM(PANEL, IFLAG); break;
    case HPL_1RING: ierr = HPL_bcast_1ring(PANEL, IFLAG); break;
    case HPL_2RING_M: ierr = HPL_bcast_2rinM(PANEL, IFLAG); break;
    case HPL_2RING: ierr = HPL_bcast_2ring(PANEL, IFLAG); break;
    case HPL_BLONG_M: ierr = HPL_bcast_blonM(PANEL, IFLAG); break;
    case HPL_BLONG: ierr = HPL_bcast_blong(PANEL, IFLAG); break;
    case HPL_IBCST: ierr = HPL_bcast_ibcst(PANEL, IFLAG); break;
    default: ierr = HPL_SUCCESS;
  }

  return (ierr);
}
