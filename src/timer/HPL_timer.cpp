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
 * ---------------------------------------------------------------------
 * Static variables
 * ---------------------------------------------------------------------
 */
static int    HPL_timer_disabled;
static double HPL_timer_cpusec[HPL_NTIMER], HPL_timer_cpustart[HPL_NTIMER],
    HPL_timer_wallsec[HPL_NTIMER], HPL_timer_wallstart[HPL_NTIMER];
/*
 * ---------------------------------------------------------------------
 * User callable functions
 * ---------------------------------------------------------------------
 */
void HPL_timer_boot(void) {
  /*
   * HPL_timer_boot (re)sets all timers to 0, and enables HPL_timer.
   */

  int i;

  HPL_timer_disabled = 0;

  for(i = 0; i < HPL_NTIMER; i++) {
    HPL_timer_cpusec[i] = HPL_timer_wallsec[i] = HPL_rzero;
    HPL_timer_cpustart[i] = HPL_timer_wallstart[i] = HPL_TIMER_STARTFLAG;
  }
}

void HPL_timer(const int I) {
  /*
   * Purpose
   * =======
   *
   * HPL_timer provides a  "stopwatch"  functionality  cpu/wall  timer  in
   * seconds.  Up to  64  separate timers can be functioning at once.  The
   * first call starts the timer,  and the second stops it.  This  routine
   * can be disenabled  by calling  HPL_timer_disable(),  so that calls to
   * the timer are ignored.  This feature can be used to make sure certain
   * sections of code do not affect timings,  even  if  they call routines
   * which have HPL_timer calls in them. HPL_timer_enable() will re-enable
   * the  timer  functionality.  One  can retrieve  the current value of a
   * timer by calling
   *
   * t0 = HPL_timer_inquire( HPL_WALL_TIME | HPL_CPU_TIME, I )
   *
   * where  I  is the timer index in  [0..64).  To  initialize  the  timer
   * functionality, one must have called HPL_timer_boot()  prior to any of
   * the functions mentioned above.
   *
   * Arguments
   * =========
   *
   * I       (global input)                const int
   *         On entry, I specifies the timer to stop/start.
   *
   * ---------------------------------------------------------------------
   */

  if(HPL_timer_disabled) return;
  /*
   * If timer has not been started, start it.  Otherwise,  stop it and add
   * interval to count
   */
  if(HPL_timer_wallstart[I] == HPL_TIMER_STARTFLAG) {
    HPL_timer_wallstart[I] = HPL_timer_walltime();
    HPL_timer_cpustart[I]  = HPL_timer_cputime();
  } else {
    HPL_timer_cpusec[I] += HPL_timer_cputime() - HPL_timer_cpustart[I];
    HPL_timer_wallsec[I] += HPL_timer_walltime() - HPL_timer_wallstart[I];
    HPL_timer_wallstart[I] = HPL_TIMER_STARTFLAG;
  }
}

void HPL_timer_enable(void) {
  /*
   * HPL_timer_enable sets it so calls to HPL_timer are not ignored.
   */

  HPL_timer_disabled = 0;
  return;
}

void HPL_timer_disable(void) {
  /*
   * HPL_timer_disable sets it so calls to HPL_timer are ignored.
   */

  HPL_timer_disabled = 1;
  return;
}

double HPL_timer_inquire(const HPL_T_TIME TMTYPE, const int I) {
  /*
   * Purpose
   * =======
   *
   * HPL_timer_inquire returns  wall- or cpu- time that has accumulated in
   * timer I.
   *
   * Arguments
   * =========
   *
   * TMTYPE  (global input)              const HPL_T_TIME
   *         On entry, TMTYPE specifies what time will be returned as fol-
   *         lows
   *            = HPL_WALL_TIME : wall clock time is returned,
   *            = HPL_CPU_TIME  : CPU time is returned (default).
   *
   * I       (global input)              const int
   *         On entry, I specifies the timer to return.
   *
   * ---------------------------------------------------------------------
   */

  double time;

  /*
   * If wall- or cpu-time are not available on this machine, return
   * HPL_TIMER_ERROR
   */
  if(TMTYPE == HPL_WALL_TIME) {
    if(HPL_timer_walltime() == HPL_TIMER_ERROR)
      time = HPL_TIMER_ERROR;
    else
      time = HPL_timer_wallsec[I];
  } else {
    if(HPL_timer_cputime() == HPL_TIMER_ERROR)
      time = HPL_TIMER_ERROR;
    else
      time = HPL_timer_cpusec[I];
  }
  return (time);
}
