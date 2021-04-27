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

void HPL_pdupdateTT(HPL_T_panel* PANEL,
                    const int    NN) {
  /*
   * Purpose
   * =======
   *
   * HPL_pdupdateNT applies the row interchanges and updates part of the
   * trailing  (using the panel PANEL) submatrix.
   *
   * Arguments
   * =========
   *
   * PANEL   (local input/output)          HPL_T_panel *
   *         On entry,  PANEL  points to the data structure containing the
   *         panel (to be updated) information.
   *
   * NN      (local input)                 const int
   *         On entry, NN specifies  the  local  number  of columns of the
   *         trailing  submatrix  to be updated  starting  at the  current
   *         position. NN must be at least zero.
   *
   * ---------------------------------------------------------------------
   */

  double *Aptr, *L1ptr, *L2ptr, *Uptr, *dpiv;
  int*    dipiv;

  int               curr, i, iroff, jb, lda, ldl2, mp, n, nb;
#define LDU n
/* ..
 * .. Executable Statements ..
 */
#ifdef HPL_DETAILED_TIMING
  HPL_ptimer(HPL_TIMING_UPDATE);
#endif
  nb  = PANEL->nb;
  jb  = PANEL->jb;
  n   = PANEL->nq;
  lda = PANEL->dlda;
  if(NN >= 0) n = Mmin(NN, n);
  /*
   * There is nothing to update, enforce the panel broadcast.
   */
  if((n <= 0) || (jb <= 0)) {
#ifdef HPL_DETAILED_TIMING
    HPL_ptimer(HPL_TIMING_UPDATE);
#endif
    return;
  }

  hipStream_t stream;
  rocblas_get_stream(handle, &stream);

  /*
   * 1 x Q case
   */
  if(PANEL->grid->nprow == 1) {
    Aptr  = PANEL->dA;
    L2ptr = PANEL->dL2;
    L1ptr = PANEL->dL1;
    ldl2  = PANEL->dldl2;

    dipiv = PANEL->dipiv;

    mp    = PANEL->mp - jb;
    iroff = PANEL->ii;

    /*
     * Update
     */
#ifdef HPL_DETAILED_TIMING
    HPL_ptimer(HPL_TIMING_LASWP);
    HPL_dlaswp00N(jb, n, Aptr, lda, dipiv);
    HPL_ptimer(HPL_TIMING_LASWP);
#else
    HPL_dlaswp00N(jb, n, Aptr, lda, dipiv);
#endif
    const double one = 1.0;
    rocblas_dtrsm(handle,
                  rocblas_side_left,
                  rocblas_fill_upper,
                  rocblas_operation_transpose,
                  rocblas_diagonal_unit,
                  jb,
                  n,
                  &one,
                  L1ptr,
                  jb,
                  Aptr,
                  lda);

#ifdef HPL_DETAILED_TIMING
    hipEventRecord(dgemmStart, stream);
#endif
    const double mone = -1.0;
    rocblas_dgemm(handle,
                  rocblas_operation_none,
                  rocblas_operation_none,
                  mp,
                  n,
                  jb,
                  &mone,
                  L2ptr,
                  ldl2,
                  Aptr,
                  lda,
                  &one,
                  Mptr(Aptr, jb, 0, lda),
                  lda);
#ifdef HPL_DETAILED_TIMING
    hipEventRecord(dgemmStop, stream);
#endif

  }
  else /* nprow > 1 ... */
  {

    curr  = (PANEL->grid->myrow == PANEL->prow ? 1 : 0);
    Aptr  = PANEL->dA;
    L2ptr = PANEL->dL2;
    L1ptr = PANEL->dL1;
    Uptr  = PANEL->dU;
    ldl2  = PANEL->dldl2;
    mp    = PANEL->mp - (curr != 0 ? jb : 0);

    /*
     * Swap:broadcast U.
     */
    HPL_pdlaswpT(PANEL, n);

    /*
     * Compute redundantly row block of U and update trailing submatrix
     */
    const double one = 1.0;
    rocblas_dtrsm(handle,
                  rocblas_side_right,
                  rocblas_fill_upper,
                  rocblas_operation_none,
                  rocblas_diagonal_unit,
                  n,
                  jb,
                  &one,
                  L1ptr,
                  jb,
                  Uptr,
                  LDU);

    /*
     * Queue finishing the update
     */
    if(curr != 0) {
#ifdef HPL_DETAILED_TIMING
      hipEventRecord(dgemmStart, stream);
#endif
      const double mone = -1.0;
      rocblas_dgemm(handle,
                    rocblas_operation_none,
                    rocblas_operation_transpose,
                    mp,
                    n,
                    jb,
                    &mone,
                    L2ptr,
                    ldl2,
                    Uptr,
                    LDU,
                    &one,
                    Mptr(Aptr, jb, 0, lda),
                    lda);
#ifdef HPL_DETAILED_TIMING
      hipEventRecord(dgemmStop, stream);
#endif
      HPL_dlatcpy_gpu(jb, n, Uptr, LDU, Aptr, lda);
    } else {
#ifdef HPL_DETAILED_TIMING
      hipEventRecord(dgemmStart, stream);
#endif
      const double mone = -1.0;
      rocblas_dgemm(handle,
                    rocblas_operation_none,
                    rocblas_operation_transpose,
                    mp,
                    n,
                    jb,
                    &mone,
                    L2ptr,
                    ldl2,
                    Uptr,
                    LDU,
                    &one,
                    Aptr,
                    lda);
#ifdef HPL_DETAILED_TIMING
      hipEventRecord(dgemmStop, stream);
#endif
    }
  }

  // PANEL->A = Mptr( PANEL->A, 0, n, lda );
  PANEL->dA = Mptr(PANEL->dA, 0, n, lda);
  PANEL->nq -= n;
  PANEL->jj += n;

#ifdef HPL_DETAILED_TIMING
  HPL_ptimer(HPL_TIMING_UPDATE);
#endif
}
