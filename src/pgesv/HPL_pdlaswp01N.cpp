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

void HPL_pdlaswp01N(HPL_T_panel* PBCST,
                    int*         IFLAG,
                    HPL_T_panel* PANEL,
                    const int    NN) {
  /*
   * Purpose
   * =======
   *
   * HPL_pdlaswp01N applies the  NB  row interchanges to  NN columns of the
   * trailing submatrix and broadcast a column panel.
   *
   * A "Spread then roll" algorithm performs  the swap :: broadcast  of the
   * row panel U at once,  resulting in a minimal communication volume  and
   * a "very good"  use of the connectivity if available.  With  P  process
   * rows  and  assuming  bi-directional links,  the  running time  of this
   * function can be approximated by:
   *
   *    (log_2(P)+(P-1)) * lat +   K * NB * LocQ(N) / bdwth
   *
   * where  NB  is the number of rows of the row panel U,  N is the global
   * number of columns being updated,  lat and bdwth  are the latency  and
   * bandwidth  of  the  network  for  double  precision real words.  K is
   * a constant in (2,3] that depends on the achieved bandwidth  during  a
   * simultaneous  message exchange  between two processes.  An  empirical
   * optimistic value of K is typically 2.4.
   *
   * Arguments
   * =========
   *
   * PBCST   (local input/output)          HPL_T_panel *
   *         On entry,  PBCST  points to the data structure containing the
   *         panel (to be broadcast) information.
   *
   * IFLAG   (local input/output)          int *
   *         On entry, IFLAG  indicates  whether or not  the broadcast has
   *         already been completed.  If not,  probing will occur, and the
   *         outcome will be contained in IFLAG on exit.
   *
   * PANEL   (local input/output)          HPL_T_panel *
   *         On entry,  PANEL  points to the data structure containing the
   *         panel information.
   *
   * NN      (local input)                 const int
   *         On entry, NN specifies  the  local  number  of columns of the
   *         trailing  submatrix  to  be swapped and broadcast starting at
   *         the current position. NN must be at least zero.
   *
   * ---------------------------------------------------------------------
   */

  double *A, *U;
  double *dA, *dU;
  int *ipID, *iplen, *ipmap, *ipmapm1, *iwork, *lindxA = NULL, *lindxAU, *permU;
  int *dlindxA     = NULL, *dlindxAU, *dpermU, *dpermU_ex;
  static int equil = -1;
  int        icurrow, *iflag, *ipA, *ipl, jb, k, lda, myrow, n, nprow;
#define LDU jb

  n  = PANEL->n;
  n  = Mmin(NN, n);
  jb = PANEL->jb;
  /*
   * Quick return if there is nothing to do
   */
  if((n <= 0) || (jb <= 0)) return;
#ifdef HPL_DETAILED_TIMING
  HPL_ptimer(HPL_TIMING_LASWP);
#endif
  /*
   * Decide whether equilibration should be performed or not
   */
  if(equil == -1) equil = PANEL->algo->equil;
  /*
   * Retrieve parameters from the PANEL data structure
   */
  nprow = PANEL->grid->nprow;
  myrow = PANEL->grid->myrow;
  A     = PANEL->A;
  U     = PANEL->U;
  dA    = PANEL->dA;
  dU    = PANEL->dU;

  iflag   = PANEL->IWORK;
  lda     = PANEL->dlda;
  icurrow = PANEL->prow;
  /*
   * Compute ipID (if not already done for this panel). lindxA and lindxAU
   * are of length at most 2*jb - iplen is of size nprow+1, ipmap, ipmapm1
   * are of size nprow,  permU is of length jb, and  this function needs a
   * workspace of size max( 2 * jb (plindx1), nprow+1(equil)):
   * 1(iflag) + 1(ipl) + 1(ipA) + 9*jb + 3*nprow + 1 + MAX(2*jb,nprow+1)
   * i.e. 4 + 9*jb + 3*nprow + max(2*jb, nprow+1);
   */
  k       = (int)((unsigned int)(jb) << 1);
  ipl     = iflag + 1;
  ipID    = ipl + 1;
  ipA     = ipID + ((unsigned int)(k) << 1);
  iplen   = ipA + 1;
  ipmap   = iplen + nprow + 1;
  ipmapm1 = ipmap + nprow;
  iwork   = ipmapm1 + nprow;

  lindxA  = PANEL->lindxA;
  lindxAU = PANEL->lindxAU;
  permU   = PANEL->permU;

  dlindxA   = PANEL->dlindxA;
  dlindxAU  = PANEL->dlindxAU;
  dpermU    = PANEL->dpermU;
  dpermU_ex = dpermU + jb;

  if(*iflag == -1) /* no index arrays have been computed so far */
  {
    HPL_pipid(PANEL, ipl, ipID);
    HPL_plindx1(PANEL,
                *ipl,
                ipID,
                ipA,
                lindxA,
                lindxAU,
                iplen,
                ipmap,
                ipmapm1,
                permU,
                iwork);
    *iflag = 1;
  } else if(*iflag == 0) /* HPL_pdlaswp00N called before: reuse ipID */
  {
    HPL_plindx1(PANEL,
                *ipl,
                ipID,
                ipA,
                lindxA,
                lindxAU,
                iplen,
                ipmap,
                ipmapm1,
                permU,
                iwork);
    *iflag = 1;
  } else if((*iflag == 1) && (equil != 0)) { /* HPL_pdlaswp01N was call before
                                                only re-compute IPLEN, IPMAP */
    HPL_plindx10(PANEL, *ipl, ipID, iplen, ipmap, ipmapm1);
    *iflag = 1;
  }

  hipStream_t stream;
  rocblas_get_stream(handle, &stream);
  /*
   * Copy into U the rows to be spread (local to icurrow)
   */
  if(myrow == icurrow) {
    HPL_dlaswp01N(*ipA, n, jb, dA, lda, dU, LDU, dlindxA, dlindxAU);
#if !defined(GPU_AWARE_MPI)
    hipMemcpy2D(U,
                LDU * sizeof(double),
                dU,
                LDU * sizeof(double),
                jb * sizeof(double),
                n,
                hipMemcpyDeviceToHost);
#endif
  }
/*
 * Spread U - optionally probe for column panel
 */
#if defined(GPU_AWARE_MPI)
  hipStreamSynchronize(stream);
  HPL_spreadN(
      PBCST, IFLAG, PANEL, HplRight, n, dU, LDU, 0, iplen, ipmap, ipmapm1);
#else
  HPL_spreadN(
      PBCST, IFLAG, PANEL, HplRight, n, U, LDU, 0, iplen, ipmap, ipmapm1);
#endif

  /*
   * Local exchange (everywhere but in process row icurrow)
   */
  if(myrow != icurrow) {
#if !defined(GPU_AWARE_MPI)
    hipMemcpy2D(dU,
                LDU * sizeof(double),
                U,
                LDU * sizeof(double),
                jb * sizeof(double),
                n,
                hipMemcpyHostToDevice);
#endif
    k = ipmapm1[myrow];
    HPL_dlaswp06N(iplen[k + 1] - iplen[k],
                  n,
                  dA,
                  lda,
                  Mptr(dU, iplen[k], 0, LDU),
                  LDU,
                  dlindxA);

#if !defined(GPU_AWARE_MPI)
    hipMemcpy2D(U,
                LDU * sizeof(double),
                dU,
                LDU * sizeof(double),
                jb * sizeof(double),
                n,
                hipMemcpyDeviceToHost);
#endif
  }
  /*
   * Equilibration
   */
  if(equil != 0)
    HPL_equil(PBCST,
              IFLAG,
              PANEL,
              HplNoTrans,
              n,
              U,
              LDU,
              iplen,
              ipmap,
              ipmapm1,
              iwork);
/*
 * Rolling phase
 */
#if defined(GPU_AWARE_MPI)
  hipStreamSynchronize(stream);
  HPL_rollN(PBCST, IFLAG, PANEL, n, dU, LDU, iplen, ipmap, ipmapm1);
#else
  HPL_rollN(PBCST, IFLAG, PANEL, n, U, LDU, iplen, ipmap, ipmapm1);
#endif
/*
 * Permute U in every process row
 */
#if !defined(GPU_AWARE_MPI)
  hipMemcpy2D(dU,
              LDU * sizeof(double),
              U,
              LDU * sizeof(double),
              jb * sizeof(double),
              n,
              hipMemcpyHostToDevice);
#endif
  HPL_dlaswp00N(jb, n, dU, LDU, dpermU);

#ifdef HPL_DETAILED_TIMING
  HPL_ptimer(HPL_TIMING_LASWP);
#endif
}
