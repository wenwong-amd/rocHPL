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
#ifndef HPL_TIMER_HPP
#define HPL_TIMER_HPP
/*
 * ---------------------------------------------------------------------
 * Include files
 * ---------------------------------------------------------------------
 */
#include "hpl_misc.hpp"

/*
 * ---------------------------------------------------------------------
 * #define macro constants
 * ---------------------------------------------------------------------
 */
#define HPL_NTIMER 64
#define HPL_TIMER_STARTFLAG 5.0
#define HPL_TIMER_ERROR -1.0
/*
 * ---------------------------------------------------------------------
 * type definitions
 * ---------------------------------------------------------------------
 */
typedef enum { HPL_WALL_TIME = 101, HPL_CPU_TIME = 102 } HPL_T_TIME;
/*
 * ---------------------------------------------------------------------
 * Function prototypes
 * ---------------------------------------------------------------------
 */
double HPL_timer_cputime(void);
double HPL_timer_walltime(void);

void   HPL_timer(const int);
void   HPL_timer_boot(void);
void   HPL_timer_enable(void);
void   HPL_timer_disable(void);
double HPL_timer_inquire(const HPL_T_TIME, const int);

#endif
/*
 * End of hpl_timer.hpp
 */
