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

#define I_SEND 0
#define I_RECV 1

int HPL_binit_blonM(HPL_T_panel* PANEL) {

  if(PANEL == NULL) { return (HPL_SUCCESS); }
  if(PANEL->grid->npcol <= 1) { return (HPL_SUCCESS); }
  /*
   * Force the copy of the panel into a contiguous buffer
   */
  HPL_copyL(PANEL);

  return (HPL_SUCCESS);
}

#define _M_BUFF_S1 (void*)(PANEL->L2)
#define _M_COUNT_S1 PANEL->len
#define _M_TYPE_S1 MPI_DOUBLE

#define _M_BUFF_S2 (void*)(PANEL->L2 + ibuf)
#define _M_COUNT_S2 lbuf
#define _M_TYPE_S2 MPI_DOUBLE

#define _M_BUFF_R1 (void*)(PANEL->L2)
#define _M_COUNT_R1 PANEL->len
#define _M_TYPE_R1 MPI_DOUBLE

#define _M_BUFF_R2 (void*)(PANEL->L2 + ibuf)
#define _M_COUNT_R2 lbuf
#define _M_TYPE_R2 MPI_DOUBLE

#define _M_ROLL_BUFF_S (void*)(PANEL->L2 + ibufS)
#define _M_ROLL_COUNT_S lbufS
#define _M_ROLL_TYPE_S MPI_DOUBLE
#define _M_ROLL_BUFF_R (void*)(PANEL->L2 + ibufR)
#define _M_ROLL_COUNT_R lbufR
#define _M_ROLL_TYPE_R MPI_DOUBLE

int HPL_bcast_blonM(HPL_T_panel* PANEL, int* IFLAG) {

  MPI_Comm comm;
  int COUNT, count, go = 1, ierr = MPI_SUCCESS, ibuf, ibufR, ibufS, dummy = 0,
                    indx, ip2 = 1, k, l, lbuf, lbufR, lbufS, mask = 1, msgid,
                    mydist, mydist2, next, npm1, npm2, partner, prev, rank,
                    root, size;

  if(PANEL == NULL) {
    *IFLAG = HPL_SUCCESS;
    return (HPL_SUCCESS);
  }
  if((size = PANEL->grid->npcol) <= 1) {
    *IFLAG = HPL_SUCCESS;
    return (HPL_SUCCESS);
  }
  /*
   * Cast phase:  root process  sends to its right neighbor,  then spread
   * the panel on the other npcol - 2 processes.  If  I  am  not the root
   * process, probe for message received.  If the message is there,  then
   * receive it. If I am just after the root process, return.  Otherwise,
   * keep spreading on those npcol - 2 processes.  Otherwise,  inform the
   * caller that the panel has still not been received.
   */
  comm  = PANEL->grid->row_comm;
  rank  = PANEL->grid->mycol;
  root  = PANEL->pcol;
  msgid = PANEL->msgid;
  prev  = MModSub1(rank, size);

  if(rank == root) {
    if(ierr == MPI_SUCCESS)
      ierr = MPI_Ssend(_M_BUFF_S1,
                       _M_COUNT_S1,
                       _M_TYPE_S1,
                       MModAdd1(rank, size),
                       msgid,
                       comm);
  } else if(prev == root) {
    /*
     * This probing mechanism causes problems when lookhead is on. Too many
     * messages are exchanged  in this virtual topology  causing  a hang on
     * some machines. It is currently disabled until a better understanding
     * is acquired.
     *
     *    ierr = MPI_Iprobe( root, msgid, comm, &go, &PANEL->status[0] );
     */
    if(ierr == MPI_SUCCESS) { /* if panel is here, proceed */
      if(go != 0) {
        if(ierr == MPI_SUCCESS)
          ierr = MPI_Recv(_M_BUFF_R1,
                          _M_COUNT_R1,
                          _M_TYPE_R1,
                          root,
                          msgid,
                          comm,
                          &PANEL->status[0]);
      } else {
        *IFLAG = HPL_KEEP_TESTING;
        return (HPL_KEEP_TESTING);
      }
    }
  }
  /*
   * if I am just after the root, exit now. The message receive  completed
   * successfully, this guy is done. If there are only 2 processes in each
   * row of processes, we are done as well.
   */
  if((prev == root) || (size == 2)) {
    *IFLAG = (ierr == MPI_SUCCESS ? HPL_SUCCESS : HPL_FAILURE);
    return (*IFLAG);
  }
  /*
   * Otherwise, proceed with broadcast -  Spread  the panel across process
   * columns
   */
  npm2  = (npm1 = size - 1) - 1;
  COUNT = PANEL->len;

  k = npm2;
  while(k > 1) {
    k >>= 1;
    ip2 <<= 1;
    mask <<= 1;
    mask++;
  }
  if(rank == root)
    mydist2 = (mydist = 0);
  else
    mydist2 = (mydist = MModSub(rank, root, size) - 1);

  indx  = ip2;
  count = COUNT / npm1;
  count = Mmax(count, 1);

  do {
    mask ^= ip2;

    if((mydist & mask) == 0) {
      lbuf = COUNT - (ibuf = indx * count);
      if(indx + ip2 < npm1) {
        l    = ip2 * count;
        lbuf = Mmin(lbuf, l);
      }

      partner = mydist ^ ip2;

      if((mydist & ip2) != 0) {
        partner = MModAdd(root, partner, size);
        if(partner != root) partner = MModAdd1(partner, size);

        if(lbuf > 0) {
          if(ierr == MPI_SUCCESS)
            ierr = MPI_Recv(_M_BUFF_R2,
                            _M_COUNT_R2,
                            _M_TYPE_R2,
                            partner,
                            msgid,
                            comm,
                            &PANEL->status[0]);
        } else /* Recv message of length zero to enable probe */
        {
          if(ierr == MPI_SUCCESS)
            ierr = MPI_Recv((void*)(&dummy),
                            0,
                            MPI_BYTE,
                            partner,
                            msgid,
                            comm,
                            &PANEL->status[0]);
        }
      } else if(partner < npm1) {
        partner = MModAdd(root, partner, size);
        if(partner != root) partner = MModAdd1(partner, size);

        if(lbuf > 0) {
          if(ierr == MPI_SUCCESS)
            ierr = MPI_Ssend(
                _M_BUFF_S2, _M_COUNT_S2, _M_TYPE_S2, partner, msgid, comm);
        } else /* Recv message of length zero to enable probe */
        {
          if(ierr == MPI_SUCCESS)
            ierr =
                MPI_Ssend((void*)(&dummy), 0, MPI_BYTE, partner, msgid, comm);
        }
      }
    }

    if(mydist2 < ip2) {
      ip2 >>= 1;
      indx -= ip2;
    } else {
      mydist2 -= ip2;
      ip2 >>= 1;
      indx += ip2;
    }

  } while(ip2 > 0);
  /*
   * Roll the pieces
   */
  prev = MModSub1(rank, size);
  if(MModSub1(prev, size) == root) prev = root;
  next = MModAdd1(rank, size);
  if(rank == root) next = MModAdd1(next, size);

  for(k = 0; k < npm2; k++) {
    l = (k >> 1);
    /*
     * Who is sending to who and how much
     */
    if(((mydist + k) & 1) != 0) {
      ibufS = (indx = MModAdd(mydist, l, npm1)) * count;
      lbufS = (indx == npm2 ? COUNT : ibufS + count);
      lbufS = Mmin(COUNT, lbufS) - ibufS;
      lbufS = Mmax(0, lbufS);

      ibufR = (indx = MModSub(mydist, l + 1, npm1)) * count;
      lbufR = (indx == npm2 ? COUNT : ibufR + count);
      lbufR = Mmin(COUNT, lbufR) - ibufR;
      lbufR = Mmax(0, lbufR);

      partner = prev;
    } else {
      ibufS = (indx = MModSub(mydist, l, npm1)) * count;
      lbufS = (indx == npm2 ? COUNT : ibufS + count);
      lbufS = Mmin(COUNT, lbufS) - ibufS;
      lbufS = Mmax(0, lbufS);

      ibufR = (indx = MModAdd(mydist, l + 1, npm1)) * count;
      lbufR = (indx == npm2 ? COUNT : ibufR + count);
      lbufR = Mmin(COUNT, lbufR) - ibufR;
      lbufR = Mmax(0, lbufR);

      partner = next;
    }
    /*
     * Exchange the messages
     */
    if(lbufS > 0) {
      if(ierr == MPI_SUCCESS)
        ierr = MPI_Issend(_M_ROLL_BUFF_S,
                          _M_ROLL_COUNT_S,
                          _M_ROLL_TYPE_S,
                          partner,
                          msgid,
                          comm,
                          &PANEL->request[0]);
    } else {
      if(ierr == MPI_SUCCESS)
        ierr = MPI_Issend((void*)(&dummy),
                          0,
                          MPI_BYTE,
                          partner,
                          msgid,
                          comm,
                          &PANEL->request[0]);
    }

    if(lbufR > 0) {
      if(ierr == MPI_SUCCESS)
        ierr = MPI_Recv(_M_ROLL_BUFF_R,
                        _M_ROLL_COUNT_R,
                        _M_ROLL_TYPE_R,
                        partner,
                        msgid,
                        comm,
                        &PANEL->status[0]);
    } else {
      if(ierr == MPI_SUCCESS)
        ierr = MPI_Recv((void*)(&dummy),
                        0,
                        MPI_BYTE,
                        partner,
                        msgid,
                        comm,
                        &PANEL->status[0]);
    }

    if(ierr == MPI_SUCCESS)
      ierr = MPI_Wait(&PANEL->request[0], &PANEL->status[0]);
  }
  /*
   * If the message was received and being forwarded,  return HPL_SUCCESS.
   * If an error occured in an MPI call, return HPL_FAILURE.
   */
  *IFLAG = (ierr == MPI_SUCCESS ? HPL_SUCCESS : HPL_FAILURE);

  return (*IFLAG);
}

int HPL_bwait_blonM(HPL_T_panel* PANEL) {

  if(PANEL == NULL) { return (HPL_SUCCESS); }
  if(PANEL->grid->npcol <= 1) { return (HPL_SUCCESS); }

  return (HPL_SUCCESS);
}
