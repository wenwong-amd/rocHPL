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
#ifndef HPL_COMM_HPP
#define HPL_COMM_HPP
/*
 * ---------------------------------------------------------------------
 * Include files
 * ---------------------------------------------------------------------
 */
#include "hpl_pmisc.hpp"
#include "hpl_panel.hpp"

/*
 * ---------------------------------------------------------------------
 * #typedefs and data structures
 * ---------------------------------------------------------------------
 */
typedef enum {
  HPL_1RING   = 401, /* Increasing ring */
  HPL_1RING_M = 402, /* Increasing ring (modified) */
  HPL_2RING   = 403, /* Increasing 2-ring */
  HPL_2RING_M = 404, /* Increasing 2-ring (modified) */
  HPL_BLONG   = 405, /* long broadcast */
  HPL_BLONG_M = 406, /* long broadcast (modified) */
  HPL_IBCST   = 407  /* MPI IBCST */
} HPL_T_TOP;
/*
 * ---------------------------------------------------------------------
 * #define macro constants
 * ---------------------------------------------------------------------
 */
#define HPL_FAILURE 0
#define HPL_SUCCESS 1
#define HPL_KEEP_TESTING 2
/*
 * ---------------------------------------------------------------------
 * comm function prototypes
 * ---------------------------------------------------------------------
 */
int  HPL_send(double*, int, int, int, MPI_Comm);
int  HPL_recv(double*, int, int, int, MPI_Comm);
int  HPL_sdrv(double*, int, int, double*, int, int, int, MPI_Comm);
int  HPL_binit(HPL_T_panel*);
int  HPL_bcast(HPL_T_panel*, int*);
int  HPL_bwait(HPL_T_panel*);
int  HPL_packL(HPL_T_panel*, const int, const int, const int);
void HPL_copyL(HPL_T_panel*);

int HPL_binit_1ring(HPL_T_panel*);
int HPL_bcast_1ring(HPL_T_panel*, int*);
int HPL_bwait_1ring(HPL_T_panel*);

int HPL_binit_1rinM(HPL_T_panel*);
int HPL_bcast_1rinM(HPL_T_panel*, int*);
int HPL_bwait_1rinM(HPL_T_panel*);

int HPL_binit_2ring(HPL_T_panel*);
int HPL_bcast_2ring(HPL_T_panel*, int*);
int HPL_bwait_2ring(HPL_T_panel*);

int HPL_binit_2rinM(HPL_T_panel*);
int HPL_bcast_2rinM(HPL_T_panel*, int*);
int HPL_bwait_2rinM(HPL_T_panel*);

int HPL_binit_blong(HPL_T_panel*);
int HPL_bcast_blong(HPL_T_panel*, int*);
int HPL_bwait_blong(HPL_T_panel*);

int HPL_binit_blonM(HPL_T_panel*);
int HPL_bcast_blonM(HPL_T_panel*, int*);
int HPL_bwait_blonM(HPL_T_panel*);

int HPL_binit_ibcst(HPL_T_panel*);
int HPL_bcast_ibcst(HPL_T_panel*, int*);
int HPL_bwait_ibcst(HPL_T_panel*);

#endif
/*
 * End of hpl_comm.hpp
 */
