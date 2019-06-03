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
#include "hpl.h"

void HPL_unroll_ipiv(const int mp, const int jb,
                     int* ipiv, int * ipiv_ex, int *upiv) {

   for(int i = 0; i < mp; i++ ) { upiv[i] = i; } //initialize ids
   for(int i = 0; i < jb; i++ ) { //swap ids
      int id = upiv[i];
      upiv[i] = upiv[ipiv[i]];
      upiv[ipiv[i]] = id;
   }

   for(int i = 0; i < jb; i++ ) { ipiv_ex[i]=-1;}

   int cnt=0;
   for(int i = jb; i < mp; i++ ) { //find swapped ids outside of panel
      if (upiv[i]<jb) {
         ipiv_ex[upiv[i]] = i;
      }
   }
}

#ifdef STDC_HEADERS
void HPL_pdpanel_SendToDevice
(
   HPL_T_panel *                    PANEL
)
#else
void HPL_pdpanel_SendToDevice
( PANEL )
   HPL_T_panel *                    PANEL;
#endif
{
   double *A, *dA;
   int jb, i, ml2;
   static int                equil=-1;
/* ..
 * .. Executable Statements ..
 */
   jb = PANEL->jb;

   if(  jb <= 0 ) return;

#ifdef ROCM

   //copy A and/or L2
   if( PANEL->grid->mycol == PANEL->pcol ) {
      A  = Mptr( PANEL->A,  0, -jb, PANEL->lda );
      dA = Mptr( PANEL->dA, 0, -jb, PANEL->lda );

      if (PANEL->mp>0)
        hipMemcpy2DAsync(dA, PANEL->lda*sizeof(double),
                          A,  PANEL->lda*sizeof(double),
                          PANEL->mp*sizeof(double), jb,
                          hipMemcpyHostToDevice, dataStream);

#ifdef HPL_COPY_L
      //L2 is its own array
      if( PANEL->grid->myrow == PANEL->prow ) {
        if ((PANEL->mp-jb)>0)
          hipMemcpy2DAsync(PANEL->dL2, PANEL->ldl2*sizeof(double),
                            Mptr( PANEL->dA, jb, -jb, PANEL->lda ),  PANEL->lda*sizeof(double),
                            (PANEL->mp-jb)*sizeof(double), jb,
                            hipMemcpyDeviceToDevice, dataStream);
      } else {
        if ((PANEL->mp)>0)
          hipMemcpy2DAsync(PANEL->dL2, PANEL->ldl2*sizeof(double),
                            Mptr( PANEL->dA, 0, -jb, PANEL->lda ),  PANEL->lda*sizeof(double),
                            (PANEL->mp)*sizeof(double), jb,
                            hipMemcpyDeviceToDevice, dataStream);
      }
#endif

      //copy L1
      hipMemcpy2DAsync(PANEL->dL1, jb*sizeof(double),
                       PANEL->L1,  jb*sizeof(double),
                       jb*sizeof(double), jb,
                       hipMemcpyHostToDevice, dataStream);

   } else {

#if !defined(GPU_AWARE_MPI)
      //L2+L1 were recieved via MPI, send them to device
      ml2 = ( PANEL->grid->myrow == PANEL->prow ? PANEL->mp - jb : PANEL->mp );
      if (ml2>0)
        hipMemcpy2DAsync(PANEL->dL2, PANEL->ldl2*sizeof(double),
                         PANEL->L2,  PANEL->ldl2*sizeof(double),
                         ml2*sizeof(double), jb,
                         hipMemcpyHostToDevice, dataStream);

      //copy L1
      hipMemcpy2DAsync(PANEL->dL1, jb*sizeof(double),
                       PANEL->L1,  jb*sizeof(double),
                       jb*sizeof(double), jb,
                       hipMemcpyHostToDevice, dataStream);
#endif
   }

   /* TODO: This needs attention for GPU Aware MPI */
   if( PANEL->grid->nprow == 1 ) {
     //unroll pivoting and send to device now
     int *ipiv     = PANEL->IWORK;
     int *ipiv_ex  = PANEL->IWORK+jb;
     int *upiv     = PANEL->IWORK2;

     for( i = 0; i < jb; i++ ) { ipiv[i] = (int)(PANEL->DPIV[i]) - PANEL->ii; } //shift
     HPL_unroll_ipiv(PANEL->mp, jb, ipiv, ipiv_ex, upiv);

     int *dipiv    = PANEL->dIWORK;
     int *dipiv_ex = PANEL->dIWORK+jb;

     hipMemcpy2DAsync(dipiv, jb*sizeof(int),
                      upiv,  jb*sizeof(int),
                      jb*sizeof(int), 1,
                      hipMemcpyHostToDevice, dataStream);
     hipMemcpy2DAsync(dipiv_ex, jb*sizeof(int),
                      ipiv_ex,  jb*sizeof(int),
                      jb*sizeof(int), 1,
                      hipMemcpyHostToDevice, dataStream);
   } else {
      if( equil == -1 ) equil = PANEL->algo->equil;

      int k = (int)((unsigned int)(jb) << 1);
      int *iflag   = PANEL->IWORK;
      int *ipl     = iflag + 1;
      int *ipID    = ipl + 1;
      int *ipA     = ipID + ((unsigned int)(k) << 1);
      int *lindxA  = ipA + 1;
      int *lindxAU = lindxA + k;
      int *iplen   = lindxAU + k;
      int *ipmap   = iplen + PANEL->grid->nprow + 1;
      int *ipmapm1 = ipmap + PANEL->grid->nprow;
      int *permU    = ipmapm1 + PANEL->grid->nprow;
      int *permU_ex = permU + jb;
      int *iwork    = permU_ex + jb;

      int *upiv     = PANEL->IWORK2;

      int *dlindxA  = PANEL->dIWORK;
      int *dlindxAU = dlindxA + k;
      int *dpermU    = dlindxAU + k;
      int *dpermU_ex = dpermU + jb;

      if( *iflag == -1 )    /* no index arrays have been computed so far */
      {
         HPL_pipid(   PANEL,  ipl, ipID );
         HPL_plindx1( PANEL, *ipl, ipID, ipA, lindxA, lindxAU, iplen,
                      ipmap, ipmapm1, permU, iwork );
         *iflag = 1;
      }
      else if( *iflag == 0 ) /* HPL_pdlaswp00N called before: reuse ipID */
      {
         HPL_plindx1( PANEL, *ipl, ipID, ipA, lindxA, lindxAU, iplen,
                      ipmap, ipmapm1, permU, iwork );
         *iflag = 1;
      }
      else if( ( *iflag == 1 ) && ( equil != 0 ) )
      {   /* HPL_pdlaswp01N was call before only re-compute IPLEN, IPMAP */
         HPL_plindx10( PANEL, *ipl, ipID, iplen, ipmap, ipmapm1 );
         *iflag = 1;
      }

      int N = Mmax(*ipA, jb);
      if (N>0) {
         hipMemcpy2DAsync(dlindxA, k*sizeof(int),
                           lindxA, k*sizeof(int),
                           N*sizeof(int), 1,
                           hipMemcpyHostToDevice, dataStream);
         hipMemcpy2DAsync(dlindxAU, k*sizeof(int),
                           lindxAU, k*sizeof(int),
                           N*sizeof(int), 1,
                           hipMemcpyHostToDevice, dataStream);
      }

      HPL_unroll_ipiv(jb, jb, permU, permU_ex, upiv);

      hipMemcpy2DAsync(dpermU, jb*sizeof(int),
                      upiv,  jb*sizeof(int),
                      jb*sizeof(int), 1,
                      hipMemcpyHostToDevice, dataStream);
      hipMemcpy2DAsync(dpermU_ex, jb*sizeof(int),
                      permU_ex,  jb*sizeof(int),
                      jb*sizeof(int), 1,
                      hipMemcpyHostToDevice, dataStream);
   }

#endif

/*
 * End of HPL_pdpanel_SendToDevice
 */
}
