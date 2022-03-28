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

int HPL_bcast_1ring(double* SBUF, int SCOUNT, int ROOT, MPI_Comm COMM) {

  int rank, size;
  MPI_Comm_rank(COMM, &rank);
  MPI_Comm_size(COMM, &size);

  if(size<= 1) return (MPI_SUCCESS);

  /*One ring exchange to rule them all*/
  int chunk_size = 512*512; //2MB

  chunk_size = std::min(chunk_size, SCOUNT);

  MPI_Request request[4];

  request[0] = MPI_REQUEST_NULL;
  request[1] = MPI_REQUEST_NULL;
  request[2] = MPI_REQUEST_NULL;
  request[3] = MPI_REQUEST_NULL;

  const int Nchunks = (SCOUNT + chunk_size-1)/chunk_size;
  const int NchunksHalf = (Nchunks+1)/2;

  const int tag=rank;
  const int next = MModAdd1(rank, size);
  const int prev = MModSub1(rank, size);

  /*Mid point of message*/
  double *SBUF0 = SBUF;
  double *SBUF1 = SBUF + NchunksHalf * chunk_size;

  double *RBUF0 = SBUF0;
  double *RBUF1 = SBUF1;

  int save_rank = rank;

  /*Shift to ROOT=0*/
  rank = MModSub(rank, ROOT, size);

  int Nsend0 = (rank==size-1) ? 0 : NchunksHalf * chunk_size;
  int Nsend1 = (rank==1)      ? 0 : SCOUNT - NchunksHalf * chunk_size;

  int Nrecv0 = (rank==0) ? 0 : NchunksHalf * chunk_size;
  int Nrecv1 = (rank==0) ? 0 : SCOUNT - NchunksHalf * chunk_size;

  int chunk=1;

  while (Nsend0>0 || Nsend1>0 || Nrecv0>0 || Nrecv1>0) {
    /*Recv from left*/
    if (rank!=0 && rank<=chunk) {
      const int n0 = std::min(Nrecv0, chunk_size);
      if (n0) MPI_Irecv(RBUF0, n0, MPI_DOUBLE, prev, prev, COMM, request+0);
      Nrecv0 -= n0;
      RBUF0 += n0;
    }

    /*Recv from right*/
    if (rank!=0 && rank>=size-chunk) {
      const int n1 = std::min(Nrecv1, chunk_size);
      if (n1) MPI_Irecv(RBUF1, n1, MPI_DOUBLE, next, next, COMM, request+1);
      Nrecv1 -= n1;
      RBUF1 += n1;
    }

    /*Send to right*/
    if (rank!=size-1 && rank<chunk) {
      const int n0 = std::min(Nsend0, chunk_size);
      if (n0) MPI_Isend(SBUF0, n0, MPI_DOUBLE, next, tag, COMM, request+2);
      Nsend0 -= n0;
      SBUF0 += n0;
    }

    /*Send to left*/
    if (rank!=1 && ( rank==0 || rank>size-chunk ) ) {
      const int n1 = std::min(Nsend1, chunk_size);
      if (n1) MPI_Isend(SBUF1, n1, MPI_DOUBLE, prev, tag, COMM, request+3);
      Nsend1 -= n1;
      SBUF1 += n1;
    }

    MPI_Waitall(4, request, MPI_STATUSES_IGNORE);

    chunk = std::min(chunk+1, size);
  }

  return MPI_SUCCESS;
}
