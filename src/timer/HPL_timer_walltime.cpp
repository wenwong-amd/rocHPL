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

#include "hpl.h"

/*
 * Purpose
 * =======
 *
 * HPL_timer_walltime returns the elapsed (wall-clock) time.
 *
 *
 * ---------------------------------------------------------------------
 */

#include <sys/time.h>
#include <sys/resource.h>

double HPL_timer_walltime(void) {
  struct timeval tp;
  static long    start = 0, startu;

  if(!start) {
    (void)gettimeofday(&tp, NULL);
    start  = tp.tv_sec;
    startu = tp.tv_usec;
    return (HPL_rzero);
  }
  (void)gettimeofday(&tp, NULL);

  return ((double)(tp.tv_sec - start) +
          ((double)(tp.tv_usec - startu) / 1000000.0));
}
