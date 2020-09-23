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

void HPL_equil(HPL_T_panel*         PBCST,
               int*                 IFLAG,
               HPL_T_panel*         PANEL,
               const enum HPL_TRANS TRANS,
               const int            N,
               double*              U,
               const int            LDU,
               int*                 IPLEN,
               const int*           IPMAP,
               const int*           IPMAPM1,
               int*                 IWORK) {
  /*
   * Purpose
   * =======
   *
   * HPL_equil equilibrates  the  local  pieces  of U, so that on exit to
   * this function, pieces of U contained in every process row are of the
   * same size. This phase makes the rolling phase optimal.  In addition,
   * this  function probes  for  the  column panel L and forwards it when
   * possible.
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
   *         panel (to be equilibrated) information.
   *
   * TRANS   (global input)                const enum HPL_TRANS
   *         On entry, TRANS specifies whether  U  is stored in transposed
   *         or non-transposed form.
   *
   * N       (local input)                 const int
   *         On entry, N  specifies the number of rows or columns of  U. N
   *         must be at least 0.
   *
   * U       (local input/output)          double *
   *         On entry,  U  is an array of dimension (LDU,*) containing the
   *         local pieces of U in each process row.
   *
   * LDU     (local input)                 const int
   *         On entry, LDU specifies the local leading dimension of U. LDU
   *         should be at least MAX(1,IPLEN[nprow]) when  U  is stored  in
   *         non-transposed form, and MAX(1,N) otherwise.
   *
   * IPLEN   (global input)                int *
   *         On entry, IPLEN is an array of dimension NPROW+1.  This array
   *         is such that IPLEN[i+1] - IPLEN[i] is the number of rows of U
   *         in process IPMAP[i].
   *
   * IPMAP   (global input)                const int *
   *         On entry, IPMAP is an array of dimension  NPROW.  This  array
   *         contains  the  logarithmic mapping of the processes. In other
   *         words, IPMAP[myrow]  is the absolute coordinate of the sorted
   *         process.
   *
   * IPMAPM1 (global input)                const int *
   *         On entry, IPMAPM1  is an array of dimension NPROW. This array
   *         contains  the inverse of the logarithmic mapping contained in
   *         IPMAP: For i in [0.. NPROCS) IPMAPM1[IPMAP[i]] = i.
   *
   * IWORK   (workspace)                   int *
   *         On entry, IWORK is a workarray of dimension NPROW+1.
   *
   * ---------------------------------------------------------------------
   */

  int i, ip, ipU, ipcur, iprow, iptgt, lastrow, left, npm1, nprow, ll, llU,
      llcur, lltgt, right, slen, smax, smin;

  if((npm1 = (nprow = PANEL->grid->nprow) - 1) <= 1) return;
  /*
   * If the current distribution of the pieces of U is already optimal for
   * the rolling phase, then return imediately.  The  optimal distribution
   * is such that ip processes have smax items and the remaining processes
   * only have smin items. Another way to check this is to verify that all
   * differences IPLEN[i+1] - IPLEN[i] are either smin or smax.
   */
  smax = ((slen = IPLEN[nprow]) + npm1) / nprow;
  ip   = slen - nprow * (smin = slen / nprow);

  iprow = 0;
  do {
    ll = IPLEN[iprow + 1] - IPLEN[iprow];
    iprow++;
  } while((iprow < nprow) && ((ll == smin) || (ll == smax)));

  if(iprow == nprow) return;
  /*
   * Now,  we are sure  the distribution of the pieces of U is not optimal
   * with respect to the rolling phase,  thus  perform  equilibration.  Go
   * through the list of processes:  Processes  that have rows that do not
   * belong to them  with respect to the optimal mapping spread them  in a
   * logarithmic fashion. To simplify a little bit the implementation, and
   * mainly the packing, a source process row spreads its data to its left
   * first, and then to its right.
   */
  IWORK[nprow] = slen;

  for(iprow = 0; iprow < nprow; iprow++) {
    llU = IPLEN[iprow + 1] - (ipU = IPLEN[iprow]);
    if(iprow < ip) {
      lltgt = smax;
      iptgt = iprow * smax;
    } else {
      lltgt = smin;
      iptgt = iprow * smin + ip;
    }

    left  = (ipU < iptgt);
    right = (iptgt + lltgt < ipU + llU);
    /*
     * If I have something to spread to either the left or the right
     */
    if((llU > 0) &&
       (left ||
        right)) { /* Figure out how much every other process should have */

      ipcur = ipU;
      llcur = llU;

      for(i = 0; i < nprow; i++) {
        if(i < ip) {
          lltgt = smax;
          iptgt = i * smax;
        } else {
          lltgt = smin;
          iptgt = i * smin + ip;
        }
        lastrow = iptgt + lltgt - 1;

        if((lastrow >= ipcur) && (llcur > 0)) {
          ll = lastrow - ipcur + 1;
          ll = Mmin(ll, llcur);
          llcur -= ll;
        } else {
          ll = 0;
        }

        IWORK[i] = ipcur;
        ipcur += ll;
        IWORK[i + 1] = ipcur;
      }
      /*
       * Equilibration phase
       */
      if(TRANS == HplNoTrans) {
        if(left) {
          HPL_spreadN(PBCST,
                      IFLAG,
                      PANEL,
                      HplLeft,
                      N,
                      U,
                      LDU,
                      iprow,
                      IWORK,
                      IPMAP,
                      IPMAPM1);
        }

        if(right) {
          HPL_spreadN(PBCST,
                      IFLAG,
                      PANEL,
                      HplRight,
                      N,
                      U,
                      LDU,
                      iprow,
                      IWORK,
                      IPMAP,
                      IPMAPM1);
        }
      } else {
        if(left) {
          HPL_spreadT(PBCST,
                      IFLAG,
                      PANEL,
                      HplLeft,
                      N,
                      U,
                      LDU,
                      iprow,
                      IWORK,
                      IPMAP,
                      IPMAPM1);
        }

        if(right) {
          HPL_spreadT(PBCST,
                      IFLAG,
                      PANEL,
                      HplRight,
                      N,
                      U,
                      LDU,
                      iprow,
                      IWORK,
                      IPMAP,
                      IPMAPM1);
        }
      }
    }
  }
  /*
   * Finally update  IPLEN  with the indexes corresponding to the new dis-
   * tribution of U - IPLEN[nprow] remained unchanged.
   */
  for(i = 0; i < nprow; i++) IPLEN[i] = (i < ip ? i * smax : i * smin + ip);
}
