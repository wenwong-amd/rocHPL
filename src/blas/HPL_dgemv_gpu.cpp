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
 * ---------------------------------------------------------------------
 */
/*
 * Include files
 */
#include "hpl.h"

#include <hip/hip_runtime.h>


#define DIM_X 64 //
#define DIM_Y 16 // GEMVN_DIM_Y must be at least 4, 8 * 8 is very slow only 40Gflop/s

__global__ void dgemv_kernel(const int m,
                             const int n,
                             const double alpha,
                             const double* __restrict__ A,
                             const int lda,
                             const double* __restrict__ x,
                             const int incx,
                             const double beta,
                             double* y,
                             const int incy)
{
    const int thread_id = threadIdx.x + threadIdx.y * blockDim.x;

    // threads are all configurated locally
    const int tx = thread_id % DIM_X;
    const int ty = thread_id / DIM_X;

    int ind;

    __shared__ double sdata[DIM_X * 4 * DIM_Y];

    double res_A[4]; // micor tile is 4 * 4
    double res_x[4];

    res_A[0] = res_x[0] = 0.0;
    res_A[1] = res_x[0] = 0.0;
    res_A[2] = res_x[0] = 0.0;
    res_A[3] = res_x[0] = 0.0;

    ind = blockIdx.x * DIM_X * 4 + tx;

    int n_tail = n % (4 * DIM_Y);
    int col    = ty * 4;

    for(col = ty * 4; col < (n - n_tail); col += 4 * DIM_Y)
    {

        if(incx >= 0)
        {
            res_x[0] = x[(col + 0) * incx];
            res_x[1] = x[(col + 1) * incx];
            res_x[2] = x[(col + 2) * incx];
            res_x[3] = x[(col + 3) * incx];
        }
        else
        {
            res_x[0] = x[(col + 1 - n) * incx];
            res_x[1] = x[(col + 2 - n) * incx];
            res_x[2] = x[(col + 3 - n) * incx];
            res_x[3] = x[(col + 4 - n) * incx];
        }

        if(ind < m)
        {
            res_A[0] += A[ind + (col + 0) * ((size_t)lda)] * res_x[0];
            res_A[0] += A[ind + (col + 1) * ((size_t)lda)] * res_x[1];
            res_A[0] += A[ind + (col + 2) * ((size_t)lda)] * res_x[2];
            res_A[0] += A[ind + (col + 3) * ((size_t)lda)] * res_x[3];
        }

        if(ind + DIM_X < m)
        {
            res_A[1] += A[ind + DIM_X + (col + 0) * ((size_t)lda)] * res_x[0];
            res_A[1] += A[ind + DIM_X + (col + 1) * ((size_t)lda)] * res_x[1];
            res_A[1] += A[ind + DIM_X + (col + 2) * ((size_t)lda)] * res_x[2];
            res_A[1] += A[ind + DIM_X + (col + 3) * ((size_t)lda)] * res_x[3];
        }

        if(ind + 2 * DIM_X < m)
        {
            res_A[2] += A[ind + 2 * DIM_X + (col + 0) * ((size_t)lda)] * res_x[0];
            res_A[2] += A[ind + 2 * DIM_X + (col + 1) * ((size_t)lda)] * res_x[1];
            res_A[2] += A[ind + 2 * DIM_X + (col + 2) * ((size_t)lda)] * res_x[2];
            res_A[2] += A[ind + 2 * DIM_X + (col + 3) * ((size_t)lda)] * res_x[3];
        }

        if(ind + 3 * DIM_X < m)
        {
            res_A[3] += A[ind + 3 * DIM_X + (col + 0) * ((size_t)lda)] * res_x[0];
            res_A[3] += A[ind + 3 * DIM_X + (col + 1) * ((size_t)lda)] * res_x[1];
            res_A[3] += A[ind + 3 * DIM_X + (col + 2) * ((size_t)lda)] * res_x[2];
            res_A[3] += A[ind + 3 * DIM_X + (col + 3) * ((size_t)lda)] * res_x[3];
        }
    }

    // if n  is not multiple of (DIM_Y * 4)
    if(n_tail > 0)
    {
        if(incx >= 0)
        {
            res_x[0] = (col < n) ? x[(col + 0) * incx] : 0;
            res_x[1] = (col + 1 < n) ? x[(col + 1) * incx] : 0;
            res_x[2] = (col + 2 < n) ? x[(col + 2) * incx] : 0;
            res_x[3] = (col + 3 < n) ? x[(col + 3) * incx] : 0;
        }
        else
        {
            res_x[0] = (col < n) ? x[(col + 1 - n) * incx] : 0;
            res_x[1] = (col + 1 < n) ? x[(col + 2 - n) * incx] : 0;
            res_x[2] = (col + 2 < n) ? x[(col + 3 - n) * incx] : 0;
            res_x[3] = (col + 3 < n) ? x[(col + 4 - n) * incx] : 0;
        }

        if(ind < m)
        {
            res_A[0] += A[ind + (col + 0) * ((size_t)lda) * (col + 0 < n)] * res_x[0];
            res_A[0] += A[ind + (col + 1) * ((size_t)lda) * (col + 1 < n)] * res_x[1];
            res_A[0] += A[ind + (col + 2) * ((size_t)lda) * (col + 2 < n)] * res_x[2];
            res_A[0] += A[ind + (col + 3) * ((size_t)lda) * (col + 3 < n)] * res_x[3];
        }

        if(ind + DIM_X < m)
        {
            res_A[1] += A[ind + DIM_X + (col + 0) * ((size_t)lda) * (col + 0 < n)] * res_x[0];
            res_A[1] += A[ind + DIM_X + (col + 1) * ((size_t)lda) * (col + 1 < n)] * res_x[1];
            res_A[1] += A[ind + DIM_X + (col + 2) * ((size_t)lda) * (col + 2 < n)] * res_x[2];
            res_A[1] += A[ind + DIM_X + (col + 3) * ((size_t)lda) * (col + 3 < n)] * res_x[3];
        }

        if(ind + 2 * DIM_X < m)
        {
            res_A[2] += A[ind + 2 * DIM_X + (col + 0) * ((size_t)lda) * (col + 0 < n)] * res_x[0];
            res_A[2] += A[ind + 2 * DIM_X + (col + 1) * ((size_t)lda) * (col + 1 < n)] * res_x[1];
            res_A[2] += A[ind + 2 * DIM_X + (col + 2) * ((size_t)lda) * (col + 2 < n)] * res_x[2];
            res_A[2] += A[ind + 2 * DIM_X + (col + 3) * ((size_t)lda) * (col + 3 < n)] * res_x[3];
        }

        if(ind + 3 * DIM_X < m)
        {
            res_A[3] += A[ind + 3 * DIM_X + (col + 0) * ((size_t)lda) * (col + 0 < n)] * res_x[0];
            res_A[3] += A[ind + 3 * DIM_X + (col + 1) * ((size_t)lda) * (col + 1 < n)] * res_x[1];
            res_A[3] += A[ind + 3 * DIM_X + (col + 2) * ((size_t)lda) * (col + 2 < n)] * res_x[2];
            res_A[3] += A[ind + 3 * DIM_X + (col + 3) * ((size_t)lda) * (col + 3 < n)] * res_x[3];
        }
    }

    sdata[tx + ty * DIM_X * 4]             = res_A[0];
    sdata[tx + DIM_X + ty * DIM_X * 4]     = res_A[1];
    sdata[tx + 2 * DIM_X + ty * DIM_X * 4] = res_A[2];
    sdata[tx + 3 * DIM_X + ty * DIM_X * 4] = res_A[3];

    __syncthreads();

    ind = blockIdx.x * DIM_X * 4 + thread_id;
    if(thread_id < DIM_X * 4)
    {
        for(int i = 1; i < DIM_Y; i++)
        {
            sdata[thread_id] += sdata[thread_id + DIM_X * 4 * i];
        }

        if(ind < m)
        {
            if(incy >= 0)
            {
                y[ind * incy] = alpha * sdata[thread_id] + beta * y[ind * incy];
            }
            else
            {
                y[(1 - m + ind) * incy] = alpha * sdata[thread_id] + beta * y[(1 - m + ind) * incy];
            }
        }
    }
}

#ifdef STDC_HEADERS
void HPL_dgemv_gpu
(
   rocblas_handle                   handle,
   const int                        M,
   const int                        N,
   const double                     ALPHA,
   const double *                   A,
   const int                        LDA,
   const double *                   X,
   const int                        INCX,
   const double                     BETA,
   double *                         Y,
   const int                        INCY
)
#else
void HPL_dgemv_gpu
( handle, M, N, ALPHA, A, LDA, X, INCX, BETA, Y, INCY )
   rocblas_handle                   handle,
   const int                        M;
   const int                        N;
   const double                     ALPHA;
   const double *                   A;
   const int                        LDA;
   const double *                   X;
   const int                        INCX;
   const double                     BETA;
   double *                         Y;
   const int                        INCY;
#endif
{
/*
 * Purpose
 * =======
 *
 * HPL_dgemv performs one of the matrix-vector operations
 *
 *     y := alpha * op( A ) * x + beta * y,
 *
 *  where op( X ) is one of
 *
 *     op( X ) = X   or   op( X ) = X^T.
 *
 * where alpha and beta are scalars, x and y are vectors and  A  is an m
 * by n matrix.
 *
 * Arguments
 * =========
 *
 * ORDER   (local input)                 const enum HPL_ORDER
 *         On entry, ORDER  specifies the storage format of the operands
 *         as follows:
 *            ORDER = HplRowMajor,
 *            ORDER = HplColumnMajor.
 *
 * TRANS   (local input)                 const enum HPL_TRANS
 *         On entry,  TRANS  specifies the  operation to be performed as
 *         follows:
 *            TRANS = HplNoTrans y := alpha*A  *x + beta*y,
 *            TRANS = HplTrans   y := alpha*A^T*x + beta*y.
 *
 * M       (local input)                 const int
 *         On entry,  M  specifies  the number of rows of  the matrix A.
 *         M must be at least zero.
 *
 * N       (local input)                 const int
 *         On entry, N  specifies the number of columns of the matrix A.
 *         N must be at least zero.
 *
 * ALPHA   (local input)                 const double
 *         On entry, ALPHA specifies the scalar alpha.   When  ALPHA  is
 *         supplied as zero then  A and X  need not be set on input.
 *
 * A       (local input)                 const double *
 *         On entry,  A  points  to an array of size equal to or greater
 *         than LDA * n.  Before  entry, the leading m by n part  of the
 *         array  A  must contain the matrix coefficients.
 *
 * LDA     (local input)                 const int
 *         On entry,  LDA  specifies  the  leading  dimension  of  A  as
 *         declared  in  the  calling  (sub) program.  LDA  must  be  at
 *         least MAX(1,m).
 *
 * X       (local input)                 const double *
 *         On entry,  X  is an incremented array of dimension  at  least
 *         ( 1 + ( n - 1 ) * abs( INCX ) )  that  contains the vector x.
 *
 * INCX    (local input)                 const int
 *         On entry, INCX specifies the increment for the elements of X.
 *         INCX must not be zero.
 *
 * BETA    (local input)                 const double
 *         On entry, BETA  specifies the scalar beta.    When  ALPHA  is
 *         supplied as zero then  Y  need not be set on input.
 *
 * Y       (local input/output)          double *
 *         On entry,  Y  is an incremented array of dimension  at  least
 *         ( 1 + ( n - 1 ) * abs( INCY ) )  that  contains the vector y.
 *         Before entry with BETA non-zero, the incremented array Y must
 *         contain the vector  y.  On exit,  Y  is  overwritten  by  the
 *         updated vector y.
 *
 * INCY    (local input)                 const int
 *         On entry, INCY specifies the increment for the elements of Y.
 *         INCY must not be zero.
 *
 * ---------------------------------------------------------------------
 */

   const int blocks = (M - 1) / (DIM_X * 4) + 1;

   dim3 gemvn_grid(blocks, 1, 1);
   dim3 gemvn_threads(DIM_X, DIM_Y, 1);

   if(0.0 == ALPHA && 1.0 == BETA)
      return;

   hipStream_t stream;
   rocblas_get_stream(handle, &stream);

   hipLaunchKernelGGL((dgemv_kernel),
                      dim3(gemvn_grid),
                      dim3(gemvn_threads),
                      0,
                      stream,
                      M,
                      N,
                      ALPHA,
                      A,
                      LDA,
                      X,
                      INCX,
                      BETA,
                      Y,
                      INCY);

/*
 * End of HPL_dgemv
 */
}
