/* ---------------------------------------------------------------------
 * -- High Performance Computing Linpack Benchmark (HPL)
 *    HPL - 2.2 - February 24, 2016
 *    Antoine P. Petitet
 *    University of Tennessee, Knoxville
 *    Innovative Computing Laboratory
 *    (C) Copyright 2000-2008 All Rights Reserved
 *
 *    Modified by: Noel Chalmers
 *    (C) 2018-2022 Advanced Micro Devices, Inc.
 *    See the rocHPL/LICENCE file for details.
 *
 *    SPDX-License-Identifier: (BSD-3-Clause)
 * ---------------------------------------------------------------------
 */

#include "hpl.hpp"

int HPL_bcast_2ring(double* SBUF, int SCOUNT, int ROOT, MPI_Comm COMM) {

  int rank, size;
  MPI_Comm_rank(COMM, &rank);
  MPI_Comm_size(COMM, &size);

  if(size <= 1) return (MPI_SUCCESS);

  /*
   * Cast phase: ROOT process  send to its right neighbor and mid-process.
   * If I am not the ROOT process,  probe for message.   If the message is
   * there,  then receive it,  and  if I am not the last process  of  both
   * rings, then forward it to the next. Otherwise, inform the caller that
   * the panel has still not been received.
   */

  int       ierr, partner, roo2;
  const int tag  = ROOT;
  const int next = MModAdd1(rank, size);
  roo2           = ((size + 1) >> 1);
  roo2           = MModAdd(ROOT, roo2, size);

  if(rank == ROOT) {
    ierr = MPI_Send(SBUF, SCOUNT, MPI_DOUBLE, next, tag, COMM);
    if((ierr == MPI_SUCCESS) && (size > 2)) {
      ierr = MPI_Send(SBUF, SCOUNT, MPI_DOUBLE, roo2, tag, COMM);
    }
  } else {
    partner = MModSub1(rank, size);
    if((partner == ROOT) || (rank == roo2)) partner = ROOT;

    ierr = MPI_Recv(
        SBUF, SCOUNT, MPI_DOUBLE, partner, tag, COMM, MPI_STATUS_IGNORE);
    if((ierr == MPI_SUCCESS) && (next != roo2) && (next != ROOT)) {
      ierr = MPI_Send(SBUF, SCOUNT, MPI_DOUBLE, next, tag, COMM);
    }
  }

  return ierr;
}
