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

#define I_SEND 0
#define I_RECV 1

void HPL_rollN(HPL_T_panel* PBCST,
               int*         IFLAG,
               HPL_T_panel* PANEL,
               const int    N,
               double*      U,
               const int    LDU,
               const int*   IPLEN,
               const int*   IPMAP,
               const int*   IPMAPM1) {
  /*
   * Purpose
   * =======
   *
   * HPL_rollN rolls the local arrays containing the local pieces of U, so
   * that on exit to this function  U  is replicated in every process row.
   * In addition, this function probe for the presence of the column panel
   * and forwards it when available.
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
   *         panel (to be rolled) information.
   *
   * N       (local input)                 const int
   *         On entry, N specifies the number of columns of  U.  N must be
   *         at least zero.
   *
   * U       (local input/output)          double *
   *         On entry,  U  is an array of dimension (LDU,*) containing the
   *         local pieces of U in each process row.
   *
   * LDU     (local input)                 const int
   *         On entry, LDU specifies the local leading dimension of U. LDU
   *         should be at least  MAX(1,IPLEN[NPROW]).
   *
   * IPLEN   (global input)                const int *
   *         On entry, IPLEN is an array of dimension NPROW+1.  This array
   *         is such that IPLEN[i+1] - IPLEN[i] is the number of rows of U
   *         in each process row.
   *
   * IPMAP   (global input)                const int *
   *         On entry, IMAP  is an array of dimension  NPROW.  This  array
   *         contains  the  logarithmic mapping of the processes. In other
   *         words,  IMAP[myrow]  is the absolute coordinate of the sorted
   *         process.
   *
   * IPMAPM1 (global input)                const int *
   *         On entry,  IMAPM1  is an array of dimension NPROW. This array
   *         contains  the inverse of the logarithmic mapping contained in
   *         IMAP: For i in [0.. NPROW) IMAPM1[IMAP[i]] = i.
   *
   * ---------------------------------------------------------------------
   */

  MPI_Datatype type[2];
  MPI_Status   status;
  MPI_Request  request;
  MPI_Comm     comm;
  int Cmsgid = MSGID_BEGIN_PFACT, ibufR, ibufS, ierr = MPI_SUCCESS, il, k, l,
      lengthR, lengthS, mydist, myrow, next, npm1, nprow, partner, prev;

  if(N <= 0) return;

  npm1  = (nprow = PANEL->grid->nprow) - 1;
  myrow = PANEL->grid->myrow;
  comm  = PANEL->grid->col_comm;
  /*
   * Rolling phase
   */
  mydist = IPMAPM1[myrow];
  prev   = IPMAP[MModSub1(mydist, nprow)];
  next   = IPMAP[MModAdd1(mydist, nprow)];

  for(k = 0; k < npm1; k++) {
    l = (int)((unsigned int)(k) >> 1);

    if(((mydist + k) & 1) != 0) {
      il      = MModAdd(mydist, l, nprow);
      lengthS = IPLEN[il + 1] - (ibufS = IPLEN[il]);
      il      = MModSub(mydist, l + 1, nprow);
      lengthR = IPLEN[il + 1] - (ibufR = IPLEN[il]);
      partner = prev;
    } else {
      il      = MModSub(mydist, l, nprow);
      lengthS = IPLEN[il + 1] - (ibufS = IPLEN[il]);
      il      = MModAdd(mydist, l + 1, nprow);
      lengthR = IPLEN[il + 1] - (ibufR = IPLEN[il]);
      partner = next;
    }

    if(lengthR > 0) {
      if(ierr == MPI_SUCCESS)
        ierr = MPI_Type_vector(N, lengthR, LDU, MPI_DOUBLE, &type[I_RECV]);
      if(ierr == MPI_SUCCESS) ierr = MPI_Type_commit(&type[I_RECV]);
      if(ierr == MPI_SUCCESS)
        ierr = MPI_Irecv(Mptr(U, ibufR, 0, LDU),
                         1,
                         type[I_RECV],
                         partner,
                         Cmsgid,
                         comm,
                         &request);
    }

    if(lengthS > 0) {
      if(ierr == MPI_SUCCESS)
        ierr = MPI_Type_vector(N, lengthS, LDU, MPI_DOUBLE, &type[I_SEND]);
      if(ierr == MPI_SUCCESS) ierr = MPI_Type_commit(&type[I_SEND]);
      if(ierr == MPI_SUCCESS)
        ierr = MPI_Send(
            Mptr(U, ibufS, 0, LDU), 1, type[I_SEND], partner, Cmsgid, comm);
      if(ierr == MPI_SUCCESS) ierr = MPI_Type_free(&type[I_SEND]);
    }

    if(lengthR > 0) {
      if(ierr == MPI_SUCCESS) ierr = MPI_Wait(&request, &status);
      if(ierr == MPI_SUCCESS) ierr = MPI_Type_free(&type[I_RECV]);
    }
    /*
     * Probe for column panel - forward it when available
     */
    if(*IFLAG == HPL_KEEP_TESTING) (void)HPL_bcast(PBCST, IFLAG);
  }

  if(ierr != MPI_SUCCESS) {
    HPL_pabort(__LINE__, "HPL_rollN", "MPI call failed");
  }
}
