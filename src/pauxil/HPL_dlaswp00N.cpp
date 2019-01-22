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
/*
 * Define default value for unrolling factor
 */

#ifdef ROCM

#include <hip/hip_runtime.h>


#if 0
#define BLOCK_SIZE 64
__global__ void dlaswp00N(const int N, const int M,
                     double* __restrict__ A,
                     const int LDA,
                     const int* __restrict__ IPIV) {

   const int n = threadIdx.x + blockDim.x*blockIdx.x;

   if (n<N)  {
      for(int i = 0; i < M; i++ ) {
         int ip = IPIV[i];

         if(i != ip) { //swap
            const double r = A[i+n*LDA];
            A[i+n*LDA] = A[ip+n*LDA];
            A[ip+n*LDA] = r;
         }
      }
   }
}
#endif

#define BLOCK_SIZE 512

__global__ void dlaswp00N(const int N, const int M,
                     double* __restrict__ A,
                     const int LDA,
                     const int* __restrict__ IPIV) {

   __shared__ double s_An_init[2048];
   __shared__ double s_An_ipiv[2048];

   const int m = threadIdx.x;
   const int n = blockIdx.x;

   //read in block column
   for (int i=m;i<M;i+=blockDim.x)
      s_An_init[i] = A[i+n*LDA];

   __syncthreads();

   //local block
   for (int i=m;i<M;i+=blockDim.x) {
      const int ip = IPIV[i];

      if (ip<M) { //local swap
         s_An_ipiv[i] = s_An_init[ip];
      } else { //non local swap
         s_An_ipiv[i] = A[ip+n*LDA];
      }
   }

   //remaining swaps in column
   for (int i=m;i<M;i+=blockDim.x) {
      const int ip_init = IPIV[i+M];
      const int ip_ipiv = IPIV[i+2*M];

      if (ip_init==ip_ipiv) break;

      if (ip_ipiv<M) { //local swap
         A[ip_init+n*LDA] = s_An_init[ip_ipiv];
      } else { //non local swap
         const double r = A[ip_init+n*LDA];
         A[ip_init+n*LDA] = A[ip_ipiv+n*LDA];
         A[ip_ipiv+n*LDA] = r;
      }
   }

   //write out local block
   for (int i=m;i<M;i+=blockDim.x)
      A[i+n*LDA] = s_An_ipiv[i];
}

#endif


#ifndef HPL_LASWP00N_DEPTH
#define    HPL_LASWP00N_DEPTH       32
#define    HPL_LASWP00N_LOG2_DEPTH   5
#endif

#ifdef STDC_HEADERS
void HPL_dlaswp00N
(
   const int                        M,
   const int                        N,
   double *                         A,
   const int                        LDA,
   const int *                      IPIV
)
#else
void HPL_dlaswp00N
( M, N, A, LDA, IPIV )
   const int                        M;
   const int                        N;
   double *                         A;
   const int                        LDA;
   const int *                      IPIV;
#endif
{
/*
 * Purpose
 * =======
 *
 * HPL_dlaswp00N performs a series of local row interchanges on a matrix
 * A. One row interchange is initiated for rows 0 through M-1 of A.
 *
 * Arguments
 * =========
 *
 * M       (local input)                 const int
 *         On entry, M specifies the number of rows of the array A to be
 *         interchanged. M must be at least zero.
 *
 * N       (local input)                 const int
 *         On entry, N  specifies  the number of columns of the array A.
 *         N must be at least zero.
 *
 * A       (local input/output)          double *
 *         On entry, A  points to an array of dimension (LDA,N) to which
 *         the row interchanges will be  applied.  On exit, the permuted
 *         matrix.
 *
 * LDA     (local input)                 const int
 *         On entry, LDA specifies the leading dimension of the array A.
 *         LDA must be at least MAX(1,M).
 *
 * IPIV    (local input)                 const int *
 *         On entry,  IPIV  is  an  array of size  M  that  contains the
 *         pivoting  information.  For  k  in [0..M),  IPIV[k]=IROFF + l
 *         implies that local rows k and l are to be interchanged.
 *
 * ---------------------------------------------------------------------
 */
/*
 * .. Local Variables ..
 */
   register double            r;
   double                     * a0, * a1;
   const int                  incA = (int)( (unsigned int)(LDA) <<
                                            HPL_LASWP00N_LOG2_DEPTH );
   int                        ip, nr, nu;
   register int               i, j;
/* ..
 * .. Executable Statements ..
 */
   if( ( M <= 0 ) || ( N <= 0 ) ) return;

#ifdef ROCM
   hipStream_t stream;
   rocblas_get_stream(handle, &stream);

   int grid_size = N;
   hipLaunchKernelGGL((dlaswp00N), dim3(grid_size), dim3(BLOCK_SIZE), 0, stream,
                                      N, M, A, LDA, IPIV);

   // for( i = 0; i < M; i++ )
   // {
   //    if( i != ( ip = IPIV[i] ) )
   //    {
   //       a0 = A + i; a1 = A + ip;
   //       rocblas_dswap(handle, N, a0, LDA, a1, LDA);
   //    }
   // }
#else
   nr = N - ( nu = (int)( ( (unsigned int)(N) >> HPL_LASWP00N_LOG2_DEPTH )
                          << HPL_LASWP00N_LOG2_DEPTH ) );

   for( j = 0; j < nu; j += HPL_LASWP00N_DEPTH, A += incA )
   {
      for( i = 0; i < M; i++ )
      {
         if( i != ( ip = IPIV[i] ) )
         {
            a0 = A + i; a1 = A + ip;

            r = *a0; *a0 = *a1; *a1 = r; a0 += LDA; a1 += LDA;
#if ( HPL_LASWP00N_DEPTH >  1 )
            r = *a0; *a0 = *a1; *a1 = r; a0 += LDA; a1 += LDA;
#endif
#if ( HPL_LASWP00N_DEPTH >  2 )
            r = *a0; *a0 = *a1; *a1 = r; a0 += LDA; a1 += LDA;
            r = *a0; *a0 = *a1; *a1 = r; a0 += LDA; a1 += LDA;
#endif
#if ( HPL_LASWP00N_DEPTH >  4 )
            r = *a0; *a0 = *a1; *a1 = r; a0 += LDA; a1 += LDA;
            r = *a0; *a0 = *a1; *a1 = r; a0 += LDA; a1 += LDA;
            r = *a0; *a0 = *a1; *a1 = r; a0 += LDA; a1 += LDA;
            r = *a0; *a0 = *a1; *a1 = r; a0 += LDA; a1 += LDA;
#endif
#if ( HPL_LASWP00N_DEPTH >  8 )
            r = *a0; *a0 = *a1; *a1 = r; a0 += LDA; a1 += LDA;
            r = *a0; *a0 = *a1; *a1 = r; a0 += LDA; a1 += LDA;
            r = *a0; *a0 = *a1; *a1 = r; a0 += LDA; a1 += LDA;
            r = *a0; *a0 = *a1; *a1 = r; a0 += LDA; a1 += LDA;
            r = *a0; *a0 = *a1; *a1 = r; a0 += LDA; a1 += LDA;
            r = *a0; *a0 = *a1; *a1 = r; a0 += LDA; a1 += LDA;
            r = *a0; *a0 = *a1; *a1 = r; a0 += LDA; a1 += LDA;
            r = *a0; *a0 = *a1; *a1 = r; a0 += LDA; a1 += LDA;
#endif
#if ( HPL_LASWP00N_DEPTH > 16 )
            r = *a0; *a0 = *a1; *a1 = r; a0 += LDA; a1 += LDA;
            r = *a0; *a0 = *a1; *a1 = r; a0 += LDA; a1 += LDA;
            r = *a0; *a0 = *a1; *a1 = r; a0 += LDA; a1 += LDA;
            r = *a0; *a0 = *a1; *a1 = r; a0 += LDA; a1 += LDA;
            r = *a0; *a0 = *a1; *a1 = r; a0 += LDA; a1 += LDA;
            r = *a0; *a0 = *a1; *a1 = r; a0 += LDA; a1 += LDA;
            r = *a0; *a0 = *a1; *a1 = r; a0 += LDA; a1 += LDA;
            r = *a0; *a0 = *a1; *a1 = r; a0 += LDA; a1 += LDA;
            r = *a0; *a0 = *a1; *a1 = r; a0 += LDA; a1 += LDA;
            r = *a0; *a0 = *a1; *a1 = r; a0 += LDA; a1 += LDA;
            r = *a0; *a0 = *a1; *a1 = r; a0 += LDA; a1 += LDA;
            r = *a0; *a0 = *a1; *a1 = r; a0 += LDA; a1 += LDA;
            r = *a0; *a0 = *a1; *a1 = r; a0 += LDA; a1 += LDA;
            r = *a0; *a0 = *a1; *a1 = r; a0 += LDA; a1 += LDA;
            r = *a0; *a0 = *a1; *a1 = r; a0 += LDA; a1 += LDA;
            r = *a0; *a0 = *a1; *a1 = r; a0 += LDA; a1 += LDA;
#endif
         }
      }
   }

   if( nr > 0 )
   {
      for( i = 0; i < M; i++ )
      {
         if( i != ( ip = IPIV[i] ) )
         {
            a0 = A + i; a1 = A + ip;
            for( j = 0; j < nr; j++, a0 += LDA, a1 += LDA )
            { r = *a0; *a0 = *a1; *a1 = r; }
         }
      }
   }
#endif
/*
 * End of HPL_dlaswp00N
 */
}
