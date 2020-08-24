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

void HPL_pdgesv(HPL_T_grid* GRID, HPL_T_palg* ALGO, HPL_T_pmat* A) {
  /*
   * Purpose
   * =======
   *
   * HPL_pdgesv factors a N+1-by-N matrix using LU factorization with row
   * partial pivoting.  The main algorithm  is the "right looking" variant
   * with  or  without look-ahead.  The  lower  triangular  factor is left
   * unpivoted and the pivots are not returned. The right hand side is the
   * N+1 column of the coefficient matrix.
   *
   * Arguments
   * =========
   *
   * GRID    (local input)                 HPL_T_grid *
   *         On entry,  GRID  points  to the data structure containing the
   *         process grid information.
   *
   * ALGO    (global input)                HPL_T_palg *
   *         On entry,  ALGO  points to  the data structure containing the
   *         algorithmic parameters.
   *
   * A       (local input/output)          HPL_T_pmat *
   *         On entry, A points to the data structure containing the local
   *         array information.
   *
   * ---------------------------------------------------------------------
   */

  if(A->n <= 0) return;

  A->info = 0;

  // if( ( ALGO->depth == 0 ) || ( GRID->npcol == 1 ) )
  // {
  // HPL_pdgesv0(  GRID, ALGO, A );
  // }
  // else
  { HPL_pdgesvK2(GRID, ALGO, A); }
  /*
   * Solve upper triangular system
   */
  if(A->info == 0) HPL_pdtrsv(GRID, A);
}
