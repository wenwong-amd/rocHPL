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

int HPL_bcast_2rinM(double* SBUF, int SCOUNT, int ROOT, MPI_Comm COMM) {

  int rank, size;
  MPI_Comm_rank(COMM, &rank);
  MPI_Comm_size(COMM, &size);

  if(size <= 1) return (MPI_SUCCESS);

  /*
   * Cast phase: ROOT process send to its two right neighbors and mid-pro-
   * cess. If I am not the ROOT process, probe for message. If the message
   * is there, then receive it. If I am not the last process of both rings
   * then forward it to the next.  Otherwise,  inform  the caller that the
   * panel has still not been received.
   */

  int       ierr, partner, roo2, next, prev;
  const int tag = ROOT;
  next          = MModAdd1(rank, size);
  prev          = MModSub1(rank, size);
  roo2          = ((size + 1) >> 1);
  roo2          = MModAdd(ROOT, roo2, size);

  if(rank == ROOT) {
    ierr = MPI_Send(SBUF, SCOUNT, MPI_DOUBLE, next, tag, COMM);

    if((ierr == MPI_SUCCESS) && (size > 2)) {
      if(MModAdd1(next, size) != roo2) {
        ierr =
            MPI_Send(SBUF, SCOUNT, MPI_DOUBLE, MModAdd1(next, size), tag, COMM);
      }
      if(ierr == MPI_SUCCESS) {
        ierr = MPI_Send(SBUF, SCOUNT, MPI_DOUBLE, roo2, tag, COMM);
      }
    }
  } else {
    prev = MModSub1(rank, size);
    if((prev == ROOT) || (rank == roo2) || (MModSub1(prev, size) == ROOT))
      partner = ROOT;
    else
      partner = prev;

    ierr = MPI_Recv(
        SBUF, SCOUNT, MPI_DOUBLE, partner, tag, COMM, MPI_STATUS_IGNORE);
    if((ierr == MPI_SUCCESS) && (prev != ROOT) && (next != roo2) &&
       (next != ROOT)) {
      ierr = MPI_Send(SBUF, SCOUNT, MPI_DOUBLE, next, tag, COMM);
    }
  }

  return ierr;
}
