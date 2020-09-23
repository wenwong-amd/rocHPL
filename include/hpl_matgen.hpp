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
#ifndef HPL_MATGEN_HPP
#define HPL_MATGEN_HPP
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
 * #define macro constants
 * ---------------------------------------------------------------------
 */
#define HPL_MULT0 1284865837
#define HPL_MULT1 1481765933
#define HPL_IADD0 1
#define HPL_IADD1 0
#define HPL_DIVFAC 2147483648.0
#define HPL_POW16 65536.0
#define HPL_HALF 0.5
/*
 * ---------------------------------------------------------------------
 * Function prototypes
 * ---------------------------------------------------------------------
 */
void   HPL_dmatgen(const int, const int, double*, const int, const int);
void   HPL_lmul(int*, int*, int*);
void   HPL_ladd(int*, int*, int*);
void   HPL_xjumpm(const int, int*, int*, int*, int*, int*, int*);
void   HPL_setran(const int, int*);
void   HPL_jumpit(int*, int*, int*, int*);
double HPL_rand(void);

#endif
/*
 * End of hpl_matgen.hpp
 */
