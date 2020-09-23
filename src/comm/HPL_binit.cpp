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

int HPL_binit(HPL_T_panel* PANEL) {
  /*
   * Purpose
   * =======
   *
   * HPL_binit initializes  a  row  broadcast.  Successful  completion  is
   * indicated by the returned error code HPL_SUCCESS.
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

  int       ierr;
  HPL_T_TOP top;

  if(PANEL->grid->npcol <= 1) return (HPL_SUCCESS);
  /*
   * Retrieve the selected virtual broadcast topology
   */
  top = PANEL->algo->btopo;

  switch(top) {
    case HPL_1RING_M: ierr = HPL_binit_1rinM(PANEL); break;
    case HPL_1RING: ierr = HPL_binit_1ring(PANEL); break;
    case HPL_2RING_M: ierr = HPL_binit_2rinM(PANEL); break;
    case HPL_2RING: ierr = HPL_binit_2ring(PANEL); break;
    case HPL_BLONG_M: ierr = HPL_binit_blonM(PANEL); break;
    case HPL_BLONG: ierr = HPL_binit_blong(PANEL); break;
    case HPL_IBCST: ierr = HPL_binit_ibcst(PANEL); break;
    default: ierr = HPL_SUCCESS;
  }

  return (ierr);
}
