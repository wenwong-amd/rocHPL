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

int HPL_allgatherv(double* BUF, const int SCOUNT, const int* RCOUNT,
                   const int* DISPL, MPI_Comm COMM) {
  /*
   * Purpose
   * =======
   *
   * HPL_allgatherv is a simple wrapper around an in-place MPI_Allgatherv.
   * Its  main  purpose is to  allow for some  experimentation / tuning
   * of this simple routine. Successful  completion  is  indicated  by
   * the  returned  error  code HPL_SUCCESS.
   *
   * Arguments
   * =========
   *
   * BUF    (local input/output)           double *
   *         On entry, on the root process BUF specifies the starting
   *         address of buffer to be gathered.
   *
   * SCOUNT  (local input)                 int
   *         On entry,  SCOUNT is an array of length SIZE specifiying 
   *         the number of  double precision entries in BUF to send to
   *         each process.
   *
   * RCOUNT  (local input)                 int
   *         On entry,  RCOUNT is an array of length SIZE specifiying
   *         the number of double precision entries in BUF to receive from
   *         each process.
   *
   * DISPL   (local input)                 int *
   *         On entry,  DISPL is an array of length SIZE specifiying the  
   *         displacement (relative to BUF) from which to place the incoming
   *         data from each process.
   *
   * COMM    (local input)                 MPI_Comm
   *         The MPI communicator identifying the communication space.
   *
   * ---------------------------------------------------------------------
   */

  roctxRangePush("HPL_Allgatherv");
  int ierr = MPI_Allgatherv(MPI_IN_PLACE,
                            SCOUNT,
                            MPI_DOUBLE,
                            BUF,
                            RCOUNT,
                            DISPL,
                            MPI_DOUBLE,
                            COMM);
  roctxRangePop();

  return ((ierr == MPI_SUCCESS ? HPL_SUCCESS : HPL_FAILURE));
}
