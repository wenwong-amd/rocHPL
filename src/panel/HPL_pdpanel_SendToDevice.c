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
/* ..
 * .. Executable Statements ..
 */
   jb = PANEL->jb;

   if(  jb <= 0 ) return;

#ifdef ROCM

   //copy A and/or L2
#ifdef HPL_COPY_L
   #error "HPL_COPY_L not supported with ROCM"
#else
   if( PANEL->grid->mycol == PANEL->pcol ) { //L2 reuses A
      A  = Mptr( PANEL->A,  0, -jb, PANEL->lda );
      dA = Mptr( PANEL->dA, 0, -jb, PANEL->lda );

      hipMemcpy2DAsync(dA, PANEL->lda*sizeof(double),
                        A,  PANEL->lda*sizeof(double),
                        PANEL->mp*sizeof(double), jb,
                        hipMemcpyHostToDevice, dataStream);

   } else {
      ml2 = ( PANEL->grid->myrow == PANEL->prow ? PANEL->mp - jb : PANEL->mp );

      if (ml2>0)
        hipMemcpy2DAsync(PANEL->dL2, PANEL->ldl2*sizeof(double),
                         PANEL->L2,  PANEL->ldl2*sizeof(double),
                         ml2*sizeof(double), jb,
                         hipMemcpyHostToDevice, dataStream);
   }
#endif

   //copy L1
   hipMemcpy2DAsync(PANEL->dL1, jb*sizeof(double),
                      PANEL->L1,  jb*sizeof(double),
                      jb*sizeof(double), jb,
                      hipMemcpyHostToDevice, dataStream);

   //unroll pivoting and send to device
   int *ipiv     = PANEL->IWORK;
   int *dipiv    = PANEL->dIWORK;
   int *ipiv_ex  = PANEL->IWORK+jb;
   int *dipiv_ex = PANEL->dIWORK+jb;

   int *upiv     = PANEL->IWORK2;

   for( i = 0; i < jb; i++ ) { ipiv[i] = (int)(PANEL->DPIV[i]) - PANEL->ii; } //shift
   for( i = 0; i < PANEL->mp; i++ ) { upiv[i] = i; } //initialize ids
   for( i = 0; i < jb; i++ ) { //swap ids
      int id = upiv[i];
      upiv[i] = upiv[ipiv[i]];
      upiv[ipiv[i]] = id;
   }

   for( i = 0; i < jb; i++ ) { ipiv_ex[i]=-1;}

   int cnt=0;
   for( i = jb; i < PANEL->mp; i++ ) { //find swapped ids outside of panel
      if (upiv[i]<jb) {
         ipiv_ex[upiv[i]] = i;
      }
   }

   hipMemcpy2DAsync(dipiv, jb*sizeof(int),
                    upiv,  jb*sizeof(int),
                    jb*sizeof(int), 1,
                    hipMemcpyHostToDevice, dataStream);
   hipMemcpy2DAsync(dipiv_ex, jb*sizeof(int),
                    ipiv_ex,  jb*sizeof(int),
                    jb*sizeof(int), 1,
                    hipMemcpyHostToDevice, dataStream);


   // for( i = 0; i < jb; i++ ) { ipiv[i] = (int)(PANEL->DPIV[i]) - PANEL->ii; }
   // hipMemcpy2DAsync(dipiv, jb*sizeof(int),
   //                  ipiv,  jb*sizeof(int),
   //                  jb*sizeof(int), 1,
   //                  hipMemcpyHostToDevice, dataStream);
#endif

/*
 * End of HPL_pdpanel_SendToDevice
 */
}
