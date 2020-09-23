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

int HPL_binit_2ring(HPL_T_panel* PANEL) {

  if(PANEL == NULL) { return (HPL_SUCCESS); }
  if(PANEL->grid->npcol <= 1) { return (HPL_SUCCESS); }
  /*
   * Force the copy of the panel into a contiguous buffer
   */
  HPL_copyL(PANEL);

  return (HPL_SUCCESS);
}

#define _M_BUFF (void*)(PANEL->L2)
#define _M_COUNT PANEL->len
#define _M_TYPE MPI_DOUBLE

int HPL_bcast_2ring(HPL_T_panel* PANEL, int* IFLAG) {

  MPI_Comm comm;
  int      ierr, go, next, msgid, partner, rank, roo2, root, size;

  if(PANEL == NULL) {
    *IFLAG = HPL_SUCCESS;
    return (HPL_SUCCESS);
  }
  if((size = PANEL->grid->npcol) <= 1) {
    *IFLAG = HPL_SUCCESS;
    return (HPL_SUCCESS);
  }
  /*
   * Cast phase: root process  send to its right neighbor and mid-process.
   * If I am not the root process,  probe for message.   If the message is
   * there,  then receive it,  and  if I am not the last process  of  both
   * rings, then forward it to the next. Otherwise, inform the caller that
   * the panel has still not been received.
   */
  rank  = PANEL->grid->mycol;
  comm  = PANEL->grid->row_comm;
  root  = PANEL->pcol;
  msgid = PANEL->msgid;
  next  = MModAdd1(rank, size);
  roo2  = ((size + 1) >> 1);
  roo2  = MModAdd(root, roo2, size);

  if(rank == root) {
    ierr = MPI_Send(_M_BUFF, _M_COUNT, _M_TYPE, next, msgid, comm);
    if((ierr == MPI_SUCCESS) && (size > 2)) {
      ierr = MPI_Send(_M_BUFF, _M_COUNT, _M_TYPE, roo2, msgid, comm);
    }
  } else {
    partner = MModSub1(rank, size);
    if((partner == root) || (rank == roo2)) partner = root;

    ierr = MPI_Iprobe(partner, msgid, comm, &go, &PANEL->status[0]);

    if(ierr == MPI_SUCCESS) {
      if(go != 0) {
        ierr = MPI_Recv(_M_BUFF,
                        _M_COUNT,
                        _M_TYPE,
                        partner,
                        msgid,
                        comm,
                        &PANEL->status[0]);
        if((ierr == MPI_SUCCESS) && (next != roo2) && (next != root)) {
          ierr = MPI_Send(_M_BUFF, _M_COUNT, _M_TYPE, next, msgid, comm);
        }
      } else {
        *IFLAG = HPL_KEEP_TESTING;
        return (*IFLAG);
      }
    }
  }
  /*
   * If the message was received and being forwarded,  return HPL_SUCCESS.
   * If an error occured in an MPI call, return HPL_FAILURE.
   */
  *IFLAG = (ierr == MPI_SUCCESS ? HPL_SUCCESS : HPL_FAILURE);

  return (*IFLAG);
}

int HPL_bwait_2ring(HPL_T_panel* PANEL) {

  if(PANEL == NULL) { return (HPL_SUCCESS); }
  if(PANEL->grid->npcol <= 1) { return (HPL_SUCCESS); }

  return (HPL_SUCCESS);
}
