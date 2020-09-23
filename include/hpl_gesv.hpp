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
#ifndef HPL_GESV_HPP
#define HPL_GESV_HPP
/*
 * ---------------------------------------------------------------------
 * Include files
 * ---------------------------------------------------------------------
 */
#include "hpl_misc.hpp"
#include "hpl_blas.hpp"
#include "hpl_auxil.hpp"

/*
 * ---------------------------------------------------------------------
 * #typedefs and data structures
 * ---------------------------------------------------------------------
 */
typedef enum {
  HPL_LEFT_LOOKING  = 301, /* Left looking lu fact variant */
  HPL_CROUT         = 302, /* Crout lu fact variant */
  HPL_RIGHT_LOOKING = 303  /* Right looking lu fact variant */
} HPL_T_FACT;
/*
 * ---------------------------------------------------------------------
 * Function prototypes
 * ---------------------------------------------------------------------
 */
void HPL_dgesv(const int,
               const int,
               const int,
               const HPL_T_FACT,
               const HPL_T_FACT,
               const int,
               double*,
               const int,
               int*);
void HPL_ipid(const int,
              double*,
              int*,
              int*,
              int*,
              int*,
              int*,
              int*,
              const int,
              const int,
              const int,
              const int,
              const int);

#endif
/*
 * End of hpl_gesv.hpp
 */
