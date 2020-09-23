/* ---------------------------------------------------------------------
 * -- High Performance Computing Linpack Benchmark (HPL)
 *    Noel Chalmers
 *    (C) 2018-2020 Advanced Micro Devices, Inc.
 *    See the rocHPL/LICENCE file for details.
 *
 *    SPDX-License-Identifier: (BSD-3-Clause)
 * ---------------------------------------------------------------------
 */

#include "hpl.hpp"

int HPL_binit_ibcst(HPL_T_panel* PANEL) {

  if(PANEL == NULL) { return (HPL_SUCCESS); }
  if(PANEL->grid->npcol <= 1) { return (HPL_SUCCESS); }

  /*
   * Force the copy of the panel into a contiguous buffer
   */
  HPL_copyL(PANEL);

  return (HPL_SUCCESS);
}

#if defined(GPU_AWARE_MPI)
#define _M_BUFF (void*)(PANEL->dL2)
#else
#define _M_BUFF (void*)(PANEL->L2)
#endif

#define _M_COUNT PANEL->len
#define _M_TYPE MPI_DOUBLE

static MPI_Request request  = MPI_REQUEST_NULL;
static MPI_Request request2 = MPI_REQUEST_NULL;

int HPL_bcast_ibcst(HPL_T_panel* PANEL, int* IFLAG) {

  MPI_Comm comm;
  int      ierr, ierr2, go, next, msgid, prev, rank, root, size;

  if(PANEL == NULL) {
    *IFLAG = HPL_SUCCESS;
    return (HPL_SUCCESS);
  }
  if((size = PANEL->grid->npcol) <= 1) {
    *IFLAG = HPL_SUCCESS;
    return (HPL_SUCCESS);
  }

  rank  = PANEL->grid->mycol;
  comm  = PANEL->grid->row_comm;
  root  = PANEL->pcol;
  msgid = PANEL->msgid;

  ierr  = MPI_Ibcast(_M_BUFF, _M_COUNT, _M_TYPE, root, comm, &request);
  ierr2 = MPI_Ibcast(PANEL->ipiv, PANEL->jb, MPI_INT, root, comm, &request2);
  /*
   * If the message was received and being forwarded,  return HPL_SUCCESS.
   * If an error occured in an MPI call, return HPL_FAILURE.
   */
  *IFLAG = (ierr == MPI_SUCCESS ? HPL_SUCCESS : HPL_FAILURE);
  *IFLAG = (ierr2 == MPI_SUCCESS ? *IFLAG : HPL_FAILURE);

  return (*IFLAG);
}

int HPL_bwait_ibcst(HPL_T_panel* PANEL) {
  int ierr1, ierr2;

  if(PANEL == NULL) { return (HPL_SUCCESS); }
  if(PANEL->grid->npcol <= 1) { return (HPL_SUCCESS); }

  ierr1 = MPI_Wait(&request, MPI_STATUS_IGNORE);
  ierr2 = MPI_Wait(&request2, MPI_STATUS_IGNORE);

  return ((ierr1 == MPI_SUCCESS
               ? (ierr2 == MPI_SUCCESS ? HPL_SUCCESS : HPL_FAILURE)
               : HPL_FAILURE));
}
