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
#ifndef HPL_TEST_HPP
#define HPL_TEST_HPP
/*
 * ---------------------------------------------------------------------
 * Include files
 * ---------------------------------------------------------------------
 */
#include "hpl_misc.hpp"
#include "hpl_blas.hpp"
#include "hpl_auxil.hpp"
#include "hpl_gesv.hpp"

#include "hpl_matgen.hpp"
#include "hpl_timer.hpp"

/*
 * ---------------------------------------------------------------------
 * Function prototypes
 * ---------------------------------------------------------------------
 */
void HPL_dinfo(FILE**,
               int*,
               int*,
               int*,
               HPL_T_FACT*,
               int*,
               int*,
               int*,
               int*,
               int*,
               HPL_T_FACT*,
               int*,
               double*,
               double*);
void HPL_dtest(FILE*,
               const int,
               const int,
               const int,
               HPL_T_FACT,
               HPL_T_FACT,
               const int,
               const double,
               const double,
               int*,
               int*,
               int*);

#endif
/*
 * End of hpl_test.hpp
 */
