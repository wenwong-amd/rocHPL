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

int HPL_scatterv(double* BUF, const int* SCOUNT, const int* DISPL,
                 const int RCOUNT, int ROOT, MPI_Comm COMM) {
  /*
   * Purpose
   * =======
   *
   * HPL_scatterv is a simple wrapper around an in-place MPI_Scatterv.
   * Its  main  purpose is to  allow for some  experimentation / tuning
   * of this simple routine. Successful  completion  is  indicated  by
   * the  returned  error  code HPL_SUCCESS.
   *
   * Arguments
   * =========
   *
   * BUF    (local input/output)           double *
   *         On entry, on the root process BUF specifies the starting
   *         address of buffer to be scattered. On non-root processes,
   *         BUF specifies the starting point of the received buffer.
   *
   * SCOUNT  (local input)                 int *
   *         On entry,  SCOUNT is an array of length SIZE specifiying 
   *         the number of  double precision entries in BUF to send to
   *         each process.
   *
   * DISPL   (local input)                 int *
   *         On entry,  DISPL is an array of length SIZE specifiying the  
   *         displacement (relative to BUF) from which to take the outgoing
   *         data to each process from the root process, and the displacement
   *         (relative to BUF) from which to receive the incoming data on
   *         each non-root process.
   *
   * RCOUNT  (local input)                 int
   *         On entry,  RCOUNT  specifies  the number of  double precision
   *         entries in BUF to be received from the ROOT process.
   *
   * ROOT    (local input)                 int
   *         On entry, ROOT specifies the rank of the origin process in
   *         the communication space defined by COMM.
   *
   * COMM    (local input)                 MPI_Comm
   *         The MPI communicator identifying the communication space.
   *
   * ---------------------------------------------------------------------
   */

  int rank, ierr;
  MPI_Comm_rank(COMM, &rank);

  roctxRangePush("HPL_Scatterv");
  if (rank==ROOT) {
    ierr = MPI_Scatterv(BUF,
                        SCOUNT,
                        DISPL,
                        MPI_DOUBLE,
                        MPI_IN_PLACE,
                        RCOUNT,
                        MPI_DOUBLE,
                        ROOT,
                        COMM);
  } else {
    ierr = MPI_Scatterv(NULL,
                        SCOUNT,
                        DISPL,
                        MPI_DOUBLE,
                        BUF,
                        RCOUNT,
                        MPI_DOUBLE,
                        ROOT,
                        COMM);
  }
  roctxRangePop();

  return ((ierr == MPI_SUCCESS ? HPL_SUCCESS : HPL_FAILURE));
}
