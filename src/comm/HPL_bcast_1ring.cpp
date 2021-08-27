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

int HPL_bcast_1ring(double* SBUF, int SCOUNT, int ROOT, MPI_Comm COMM) {

  int rank, size;
  MPI_Comm_rank(COMM, &rank);
  MPI_Comm_size(COMM, &size);

  if(size<= 1) return (MPI_SUCCESS);

  /*
   * Cast phase:  If I am the ROOT process, start spreading the panel.  If
   * I am not the ROOT process, probe for message. If the message is here,
   * then receive it, and  if I am not the last process of the ring, then
   * forward it to the next.  Otherwise, inform the caller that the panel
   * has still not been received.
   */
  const int tag=ROOT;
  const int next = MModAdd1(rank, size);
  const int prev = MModSub1(rank, size);

  int      ierr;

  if(rank == ROOT) {
    ierr = MPI_Send(SBUF, SCOUNT, MPI_DOUBLE, MModAdd1(rank, size), tag, COMM);
  } else {
    ierr = MPI_Recv(SBUF, SCOUNT, MPI_DOUBLE, prev, tag, COMM, MPI_STATUS_IGNORE);
    if((ierr == MPI_SUCCESS) && (next != ROOT)) {
      ierr = MPI_Send(SBUF, SCOUNT, MPI_DOUBLE, next, tag, COMM);
    }
  }

  return ierr;
}
