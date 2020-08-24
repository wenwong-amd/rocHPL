/*
 * -- High Performance Computing Linpack Benchmark (HPL)
 *    HPL - 2.2 - February 24, 2016
 *    Antoine P. Petitet
 *    University of Tennessee, Knoxville
 *    Innovative Computing Laboratory
 *    (C) Copyright 2000-2008 All Rights Reserved
 *
 * -- Copyright notice and Licensing terms:
 *
 * Redistribution  and  use in  source and binary forms, with or without
 * modification, are  permitted provided  that the following  conditions
 * are met:
 *
 * 1. Redistributions  of  source  code  must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce  the above copyright
 * notice, this list of conditions,  and the following disclaimer in the
 * documentation and/or other materials provided with the distribution.
 *
 * 3. All  advertising  materials  mentioning  features  or  use of this
 * software must display the following acknowledgement:
 * This  product  includes  software  developed  at  the  University  of
 * Tennessee, Knoxville, Innovative Computing Laboratory.
 *
 * 4. The name of the  University,  the name of the  Laboratory,  or the
 * names  of  its  contributors  may  not  be used to endorse or promote
 * products  derived   from   this  software  without  specific  written
 * permission.
 *
 * -- Disclaimer:
 *
 * THIS  SOFTWARE  IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES,  INCLUDING,  BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE UNIVERSITY
 * OR  CONTRIBUTORS  BE  LIABLE FOR ANY  DIRECT,  INDIRECT,  INCIDENTAL,
 * SPECIAL,  EXEMPLARY,  OR  CONSEQUENTIAL DAMAGES  (INCLUDING,  BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA OR PROFITS; OR BUSINESS INTERRUPTION)  HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT,  STRICT LIABILITY,  OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#ifndef HPL_BLAS_H
#define HPL_BLAS_H
/*
 * ---------------------------------------------------------------------
 * Include files
 * ---------------------------------------------------------------------
 */
#include "hpl_misc.h"
#include <rocblas.h>

extern rocblas_handle handle;
extern hipStream_t    computeStream;
extern hipStream_t    dataStream;

extern hipEvent_t panelUpdate;
extern hipEvent_t panelCopy;

extern hipEvent_t dlaswpStart, dlaswpStop;
extern hipEvent_t dtrsmStart, dtrsmStop;
extern hipEvent_t dgemmStart, dgemmStop;

void HPL_dgemv_gpu(rocblas_handle,
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
 * hpl_blas.h
 */
