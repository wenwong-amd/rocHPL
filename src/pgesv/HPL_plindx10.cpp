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

void HPL_plindx10(HPL_T_panel* PANEL,
                  const int    K,
                  const int*   IPID,
                  int*         IPLEN,
                  int*         IPMAP,
                  int*         IPMAPM1) {
  /*
   * Purpose
   * =======
   *
   * HPL_plindx10 computes  three arrays  IPLEN,  IPMAP  and  IPMAPM1  that
   * contain the logarithmic mapping information for the spreading phase.
   *
   * Arguments
   * =========
   *
   * PANEL   (local input/output)          HPL_T_panel *
   *         On entry,  PANEL  points to the data structure containing the
   *         panel information.
   *
   * K       (global input)                const int
   *         On entry, K specifies the number of entries in IPID.  K is at
   *         least 2*N, and at most 4*N.
   *
   * IPID    (global input)                const int *
   *         On entry,  IPID  is an array of length K. The first K entries
   *         of that array contain the src and final destination resulting
   *         from the application of the interchanges.
   *
   * IPLEN   (global output)               int *
   *         On entry, IPLEN  is an array of dimension NPROW + 1. On exit,
   *         this array is such that  IPLEN[i]  is the number of rows of A
   *         in the processes  before process IMAP[i] after the sort, with
   *         the convention that IPLEN[nprow] is the total number of rows.
   *         In other words,  IPLEN[i+1] - IPLEN[i] is the local number of
   *         rows of  A  that should be moved for each process.  IPLEN  is
   *         such that the number of rows of the source process row can be
   *         computed as IPLEN[1] - IPLEN[0], and the remaining entries of
   *         this  array are sorted  so  that  the quantities IPLEN[i+1] -
   *         IPLEN[i] are logarithmically sorted.
   *
   * IPMAP   (global output)               int *
   *         On entry, IPMAP is an array of dimension NPROW. On exit, this
   *         array contains  the logarithmic mapping of the processes.  In
   *         other words, IPMAP[myrow] is the corresponding sorted process
   *         coordinate.
   *
   * IPMAPM1 (global output)               int *
   *         On entry, IPMAPM1  is an array of dimension NPROW.  On  exit,
   *         this  array  contains  the inverse of the logarithmic mapping
   *         contained  in  IPMAP:  IPMAPM1[ IPMAP[i] ] = i,  for all i in
   *         [0.. NPROW)
   *
   * ---------------------------------------------------------------------
   */

  int dst, dstrow, i, ia, icurrow, jb, nb, nprow, src, srcrow;

  nprow   = PANEL->grid->nprow;
  jb      = PANEL->jb;
  nb      = PANEL->nb;
  ia      = PANEL->ia;
  icurrow = PANEL->prow;
  /*
   * Compute  redundantly  the local number of rows  that each process has
   * and that belong to U in IPLEN[1 .. nprow+1]
   */
  for(i = 0; i <= nprow; i++) IPLEN[i] = 0;

  for(i = 0; i < K; i += 2) {
    src = IPID[i];
    Mindxg2p(src, nb, nb, srcrow, 0, nprow);
    if(srcrow == icurrow) {
      dst = IPID[i + 1];
      Mindxg2p(dst, nb, nb, dstrow, 0, nprow);
      if((dstrow != srcrow) || (dst - ia < jb)) IPLEN[dstrow + 1]++;
    }
  }
  /*
   * Logarithmic sort of the processes - compute IPMAP, IPLEN and IPMAPM1
   * (the inverse of IPMAP)
   */
  HPL_logsort(nprow, icurrow, IPLEN, IPMAP, IPMAPM1);
}
