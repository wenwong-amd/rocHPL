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
#ifndef HPL_BLAS_HPP
#define HPL_BLAS_HPP
/*
 * ---------------------------------------------------------------------
 * Include files
 * ---------------------------------------------------------------------
 */
#include "hpl_misc.hpp"
#include <rocblas.h>

extern rocblas_handle handle;
extern hipStream_t    computeStream;
extern hipStream_t    dataStream;

extern hipEvent_t swapStartEvent, swapUCopyEvent, swapWCopyEvent;

extern hipEvent_t panelUpdate;
extern hipEvent_t panelCopy;

extern hipEvent_t dlaswpStart, dlaswpStop;
extern hipEvent_t dtrsmStart, dtrsmStop;
extern hipEvent_t dgemmStart, dgemmStop;

#if __cplusplus
extern "C" {
#endif

/*
 * ---------------------------------------------------------------------
 * typedef definitions
 * ---------------------------------------------------------------------
 */
enum HPL_ORDER { HplRowMajor = 101, HplColumnMajor = 102 };
enum HPL_TRANS { HplNoTrans = 111, HplTrans = 112, HplConjTrans = 113 };
enum HPL_UPLO { HplUpper = 121, HplLower = 122 };
enum HPL_DIAG { HplNonUnit = 131, HplUnit = 132 };
enum HPL_SIDE { HplLeft = 141, HplRight = 142 };

/*
 * ---------------------------------------------------------------------
 * #define macro constants
 * ---------------------------------------------------------------------
 */
#define CBLAS_INDEX int

#define CBLAS_ORDER HPL_ORDER
#define CblasRowMajor HplRowMajor
#define CblasColMajor HplColMajor

#define CBLAS_TRANSPOSE HPL_TRANS
#define CblasNoTrans HplNoTrans
#define CblasTrans HplTrans
#define CblasConjTrans HplConjTrans

#define CBLAS_UPLO HPL_UPLO
#define CblasUpper HplUpper
#define CblasLower HplLower

#define CBLAS_DIAG HPL_DIAG
#define CblasNonUnit HplNonUnit
#define CblasUnit HplUnit

#define CBLAS_SIDE HPL_SIDE
#define CblasLeft HplLeft
#define CblasRight HplRight
/*
 * ---------------------------------------------------------------------
 * CBLAS Function prototypes
 * ---------------------------------------------------------------------
 */
CBLAS_INDEX cblas_idamax(const int, const double*, const int);
void        cblas_dswap(const int, double*, const int, double*, const int);
void cblas_dcopy(const int, const double*, const int, double*, const int);
void cblas_daxpy(const int,
                 const double,
                 const double*,
                 const int,
                 double*,
                 const int);
void cblas_dscal(const int, const double, double*, const int);

void cblas_dgemv(const enum CBLAS_ORDER,
                 const enum CBLAS_TRANSPOSE,
                 const int,
                 const int,
                 const double,
                 const double*,
                 const int,
                 const double*,
                 const int,
                 const double,
                 double*,
                 const int);

void cblas_dger(const enum CBLAS_ORDER,
                const int,
                const int,
                const double,
                const double*,
                const int,
                const double*,
                const int,
                double*,
                const int);
void cblas_dtrsv(const enum CBLAS_ORDER,
                 const enum CBLAS_UPLO,
                 const enum CBLAS_TRANSPOSE,
                 const enum CBLAS_DIAG,
                 const int,
                 const double*,
                 const int,
                 double*,
                 const int);

void cblas_dgemm(const enum CBLAS_ORDER,
                 const enum CBLAS_TRANSPOSE,
                 const enum CBLAS_TRANSPOSE,
                 const int,
                 const int,
                 const int,
                 const double,
                 const double*,
                 const int,
                 const double*,
                 const int,
                 const double,
                 double*,
                 const int);
void cblas_dtrsm(const enum CBLAS_ORDER,
                 const enum CBLAS_SIDE,
                 const enum CBLAS_UPLO,
                 const enum CBLAS_TRANSPOSE,
                 const enum CBLAS_DIAG,
                 const int,
                 const int,
                 const double,
                 const double*,
                 const int,
                 double*,
                 const int);
/*
 * ---------------------------------------------------------------------
 * HPL C BLAS macro definition
 * ---------------------------------------------------------------------
 */
#define HPL_dswap cblas_dswap
#define HPL_dcopy cblas_dcopy
#define HPL_daxpy cblas_daxpy
#define HPL_dscal cblas_dscal
#define HPL_idamax cblas_idamax

#define HPL_dgemv cblas_dgemv
#define HPL_dtrsv cblas_dtrsv
#define HPL_dger cblas_dger

#define HPL_dgemm cblas_dgemm
#define HPL_dtrsm cblas_dtrsm

#if __cplusplus
}
#endif

#endif
/*
 * hpl_blas.hpp
 */
