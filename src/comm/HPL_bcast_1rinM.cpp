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

int HPL_bcast_1rinM(double* SBUF, int SCOUNT, int ROOT, MPI_Comm COMM) {

  int rank, size;
  MPI_Comm_rank(COMM, &rank);
  MPI_Comm_size(COMM, &size);

  if(size <= 1) return (MPI_SUCCESS);

  /*
   * Cast phase:  If I am the ROOT process,  then  send message to its two
   * next neighbors. Otherwise, probe for message. If the message is here,
   * then receive it,   and  if I am not the last process of the ring,  or
   * just after the ROOT process, then forward it to the next.  Otherwise,
   * inform the caller that the panel has still not been received.
   */
  int       ierr, partner, next, prev;
  const int tag = ROOT;
  next          = MModAdd1(rank, size);
  prev          = MModSub1(rank, size);

  if(rank == ROOT) {
    ierr = MPI_Send(SBUF, SCOUNT, MPI_DOUBLE, next, tag, COMM);
    if((ierr == MPI_SUCCESS) && (size > 2)) {
      ierr =
          MPI_Send(SBUF, SCOUNT, MPI_DOUBLE, MModAdd1(next, size), tag, COMM);
    }
  } else {

    if((size > 2) && (MModSub1(prev, size) == ROOT))
      partner = ROOT;
    else
      partner = prev;

    ierr = MPI_Recv(
        SBUF, SCOUNT, MPI_DOUBLE, partner, tag, COMM, MPI_STATUS_IGNORE);
    if((ierr == MPI_SUCCESS) && (prev != ROOT) && (next != ROOT)) {
      ierr = MPI_Send(SBUF, SCOUNT, MPI_DOUBLE, next, tag, COMM);
    }
  }

  return ierr;
}
