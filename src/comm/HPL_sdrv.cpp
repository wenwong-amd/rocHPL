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

int HPL_sdrv(double*  SBUF,
             int      SCOUNT,
             int      STAG,
             double*  RBUF,
             int      RCOUNT,
             int      RTAG,
             int      PARTNER,
             MPI_Comm COMM) {
  /*
   * Purpose
   * =======
   *
   * HPL_sdrv is a simple wrapper around MPI_Sendrecv. Its main purpose is
   * to allow for some experimentation and tuning of this simple function.
   * Messages  of  length  less than  or  equal to zero  are not sent  nor
   * received.  Successful completion  is  indicated by the returned error
   * code HPL_SUCCESS.
   *
   * Arguments
   * =========
   *
   * SBUF    (local input)                 double *
   *         On entry, SBUF specifies the starting address of buffer to be
   *         sent.
   *
   * SCOUNT  (local input)                 int
   *         On entry,  SCOUNT  specifies  the number  of double precision
   *         entries in SBUF. SCOUNT must be at least zero.
   *
   * STAG    (local input)                 int
   *         On entry,  STAG  specifies the message tag to be used for the
   *         sending communication operation.
   *
   * RBUF    (local output)                double *
   *         On entry, RBUF specifies the starting address of buffer to be
   *         received.
   *
   * RCOUNT  (local input)                 int
   *         On entry,  RCOUNT  specifies  the number  of double precision
   *         entries in RBUF. RCOUNT must be at least zero.
   *
   * RTAG    (local input)                 int
   *         On entry,  RTAG  specifies the message tag to be used for the
   *         receiving communication operation.
   *
   * PARTNER (local input)                 int
   *         On entry,  PARTNER  specifies  the rank of the  collaborative
   *         process in the communication space defined by COMM.
   *
   * COMM    (local input)                 MPI_Comm
   *         The MPI communicator identifying the communication space.
   *
   * ---------------------------------------------------------------------
   */

  MPI_Request request;
  MPI_Status  status;
  int         ierr;

  if(RCOUNT > 0) {
    if(SCOUNT > 0) {
      /*
       * Post asynchronous receive
       */
      ierr = MPI_Irecv(
          (void*)(RBUF), RCOUNT, MPI_DOUBLE, PARTNER, RTAG, COMM, &request);
      /*
       * Blocking send
       */
      if(ierr == MPI_SUCCESS)
        ierr = MPI_Send((void*)(SBUF), SCOUNT, MPI_DOUBLE, PARTNER, STAG, COMM);
      /*
       * Wait for the receive to complete
       */
      if(ierr == MPI_SUCCESS) ierr = MPI_Wait(&request, &status);

    } else {
      /*
       * Blocking receive
       */
      ierr = MPI_Recv(
          (void*)(RBUF), RCOUNT, MPI_DOUBLE, PARTNER, RTAG, COMM, &status);
    }
  } else if(SCOUNT > 0) {
    /*
     * Blocking send
     */
    ierr = MPI_Send((void*)(SBUF), SCOUNT, MPI_DOUBLE, PARTNER, STAG, COMM);
  } else {
    ierr = MPI_SUCCESS;
  }

  return ((ierr == MPI_SUCCESS ? HPL_SUCCESS : HPL_FAILURE));
}
