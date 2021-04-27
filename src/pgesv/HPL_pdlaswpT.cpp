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

void HPL_pdlaswpT(HPL_T_panel* PANEL, const int NN) {
  /*
   * Purpose
   * =======
   *
   * HPL_pdlaswpT begins the  NB  row interchanges to  NN columns of the
   * trailing submatrix and broadcast a column panel. The rows needed for
   * the row interchanges are packed into U.
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
   * The scattered rows  of  A  are recieved into a work array W.  The
   * row offsets in  A  of the source rows  are specified by LINDXA.  The
   * destination of those rows are specified by  LINDXAU.  A
   * positive value of LINDXAU indicates that the array  destination is U,
   * and A otherwise. Rows of A are stored as columns in U.
   *
   * Arguments
   * =========
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
  /*
   * .. Local Variables ..
   */
  double *U, *W;
  double *dA, *dU, *dW;
  int *ipID, *iplen, *ipcounts, *ipoffsets, *iwork, *lindxA = NULL, *lindxAU, *permU;
  int *dlindxA     = NULL, *dlindxAU, *dpermU, *dpermU_ex;
  int        icurrow, *iflag, *ipA, *ipl, jb, k, lda, myrow, n, nprow, LDU, LDW;

  /* ..
   * .. Executable Statements ..
   */
  n  = PANEL->n;
  n  = Mmin(NN, n);
  jb = PANEL->jb;
  /*
   * Quick return if there is nothing to do
   */
  if((n <= 0) || (jb <= 0)) return;

  /*
   * Retrieve parameters from the PANEL data structure
   */
  nprow = PANEL->grid->nprow;
  myrow = PANEL->grid->myrow;
  iflag = PANEL->IWORK;

  MPI_Comm comm  = PANEL->grid->col_comm;

  //quick return if we're 1xQ
  if (nprow==1) return;

  dA      = PANEL->dA;
  lda     = PANEL->dlda;
  icurrow = PANEL->prow;

  U       = PANEL->U;
  W       = PANEL->W;
  dU      = PANEL->dU;
  dW      = PANEL->dW;
#define LDU n
#define LDW n

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
  ipcounts  = iplen + nprow + 1;
  ipoffsets = ipcounts + nprow;
  iwork     = ipoffsets + nprow;

  lindxA  = PANEL->lindxA;
  lindxAU = PANEL->lindxAU;
  permU   = PANEL->permU;

  dlindxA   = PANEL->dlindxA;
  dlindxAU  = PANEL->dlindxAU;
  dpermU    = PANEL->dpermU;
  dpermU_ex = dpermU + jb;

  if(*iflag == -1) /* no index arrays have been computed so far */
  {
#ifdef GPU_AWARE_MPI
    //get the ipivs on the host after the Bcast
    if(PANEL->grid->mycol != PANEL->pcol) {
      hipMemcpy2DAsync(PANEL->ipiv,
                       PANEL->jb * sizeof(int),
                       PANEL->dipiv,
                       PANEL->jb * sizeof(int),
                       PANEL->jb * sizeof(int),
                       1,
                       hipMemcpyDeviceToHost,
                       dataStream);
    }
    hipStreamSynchronize(dataStream);
#endif

    //compute spreading info
    HPL_pipid(PANEL, ipl, ipID);
    HPL_plindx(PANEL,
               *ipl,
               ipID,
               ipA,
               lindxA,
               lindxAU,
               iplen,
               permU,
               iwork);
    *iflag = 1;
  }

  /* Set MPI message counts and offsets */
  ipcounts[0]  = (iplen[1]-iplen[0])*n;
  ipoffsets[0] = 0;

  for (int i=1;i<nprow;++i) {
    ipcounts[i]  = (iplen[i+1]-iplen[i])*n;
    ipoffsets[i] = ipcounts[i-1] + ipoffsets[i-1];
  }

  /*
   * For i in [0..2*jb),  lindxA[i] is the offset in A of a row that ulti-
   * mately goes to U( :, lindxAU[i] ).  In each rank, we directly pack
   * into U, otherwise we pack into workspace. The  first
   * entry of each column packed in workspace is in fact the row or column
   * offset in U where it should go to.
   */

  if(myrow == icurrow) {
    // copy needed rows of A into U
    HPL_dlaswp01T(*ipA, n, dA, lda, dU, LDU, dlindxA, dlindxAU);

    // record when packing completes
    hipEventRecord(swapStartEvent, computeStream);

#if defined(GPU_AWARE_MPI)
    // swap rows local to A on device
    HPL_dlaswp02T(*ipA, n, dA, lda, dlindxA, dlindxAU);

    hipEventSynchronize(swapStartEvent);

    //send rows to other ranks
    MPI_Scatterv(dU, ipcounts, ipoffsets, MPI_DOUBLE,
                 MPI_IN_PLACE, ipcounts[myrow], MPI_DOUBLE, icurrow, comm);

    //All gather dU
    MPI_Allgatherv(MPI_IN_PLACE, ipcounts[myrow], MPI_DOUBLE,
                   dU, ipcounts, ipoffsets, MPI_DOUBLE, comm);
#else
    // Copy U to host
    hipStreamWaitEvent(dataStream, swapStartEvent, 0);
    hipMemcpy2DAsync(U,
                     LDU * sizeof(double),
                     dU,
                     LDU * sizeof(double),
                     n * sizeof(double),
                     jb,
                     hipMemcpyDeviceToHost,
                     dataStream);

    // swap rows local to A on device
    HPL_dlaswp02T(*ipA, n, dA, lda, dlindxA, dlindxAU);

    hipStreamSynchronize(dataStream);

    //send rows to other ranks
    MPI_Scatterv(U, ipcounts, ipoffsets, MPI_DOUBLE,
                 MPI_IN_PLACE, ipcounts[myrow], MPI_DOUBLE, icurrow, comm);

    //All gather U
    MPI_Allgatherv(MPI_IN_PLACE, ipcounts[myrow], MPI_DOUBLE,
                   U, ipcounts, ipoffsets, MPI_DOUBLE, comm);

    //send U to device
    hipMemcpy2DAsync(dU,
                     LDU * sizeof(double),
                     U,
                     LDU * sizeof(double),
                     n * sizeof(double),
                     jb,
                     hipMemcpyHostToDevice,
                     dataStream);
    hipEventRecord(swapUCopyEvent, dataStream);
    hipStreamWaitEvent(computeStream, swapUCopyEvent, 0);
#endif

  } else {

    //queue copy kernel for needed rows from A into U(:, iplen[myrow])
    HPL_dlaswp03T(iplen[myrow + 1] - iplen[myrow], n, dA, lda,
                  Mptr(dU, 0, iplen[myrow], LDU), LDU, dlindxA);

    // record when packing completes
    hipEventRecord(swapStartEvent, computeStream);

#if defined(GPU_AWARE_MPI)
    //receive rows from icurrow into dW
    MPI_Scatterv(NULL, ipcounts, ipoffsets, MPI_DOUBLE,
                 dW, ipcounts[myrow], MPI_DOUBLE, icurrow, comm);

    // Queue inserting recieved rows in W into A on device
    HPL_dlaswp04T(iplen[myrow + 1] - iplen[myrow], n,
                  dA, lda, dW, LDW, dlindxA);

    //wait for dU to be ready
    hipEventSynchronize(swapStartEvent);

    //All gather dU
    MPI_Allgatherv(MPI_IN_PLACE, ipcounts[myrow], MPI_DOUBLE,
                   dU, ipcounts, ipoffsets, MPI_DOUBLE, comm);
#else

    // Copy my U piece to host
    hipStreamWaitEvent(dataStream, swapStartEvent, 0);
    hipMemcpy2DAsync(Mptr(U, 0, iplen[myrow], LDU),
                     LDU * sizeof(double),
                     Mptr(dU, 0, iplen[myrow], LDU),
                     LDU * sizeof(double),
                     n * sizeof(double),
                     iplen[myrow + 1] - iplen[myrow],
                     hipMemcpyDeviceToHost,
                     dataStream);
    hipEventRecord(swapUCopyEvent, dataStream);

    //receive rows from icurrow into W
    MPI_Scatterv(NULL, ipcounts, ipoffsets, MPI_DOUBLE,
                 W, ipcounts[myrow], MPI_DOUBLE, icurrow, comm);

    //wait for U to be ready
    hipEventSynchronize(swapUCopyEvent);

    // Copy recieved W piece to device
    hipMemcpy2DAsync(dW,
                     LDW * sizeof(double),
                     W,
                     LDW * sizeof(double),
                     n * sizeof(double),
                     iplen[myrow + 1] - iplen[myrow],
                     hipMemcpyHostToDevice,
                     dataStream);
    hipEventRecord(swapWCopyEvent, dataStream);

    // Queue inserting recieved rows in W into A on device
    hipStreamWaitEvent(computeStream, swapWCopyEvent, 0);
    HPL_dlaswp04T(iplen[myrow + 1] - iplen[myrow], n,
                  dA, lda, dW, LDW, dlindxA);

    //All gather U
    MPI_Allgatherv(MPI_IN_PLACE, ipcounts[myrow], MPI_DOUBLE,
                   U, ipcounts, ipoffsets, MPI_DOUBLE, comm);

    //send U to device
    hipMemcpy2DAsync(dU,
                     LDU * sizeof(double),
                     U,
                     LDU * sizeof(double),
                     n * sizeof(double),
                     jb,
                     hipMemcpyHostToDevice,
                     dataStream);
    hipEventRecord(swapUCopyEvent, dataStream);
    hipStreamWaitEvent(computeStream, swapUCopyEvent, 0);
#endif
  }

  /*
   * Permute U in every process row
   */
  HPL_dlaswp10N(n, jb, dU, LDU, dpermU);

  /*
   * End of HPL_pdlaswpT
   */
}
