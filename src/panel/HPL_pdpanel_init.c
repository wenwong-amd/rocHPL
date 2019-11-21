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

#ifdef HPL_NO_MPI_DATATYPE  /* The user insists to not use MPI types */
#ifndef HPL_COPY_L       /* and also want to avoid the copy of L ... */
#define HPL_COPY_L   /* well, sorry, can not do that: force the copy */
#endif
#endif

#ifdef STDC_HEADERS
void HPL_pdpanel_init
(
   HPL_T_grid *                     GRID,
   HPL_T_palg *                     ALGO,
   const int                        M,
   const int                        N,
   const int                        JB,
   HPL_T_pmat *                     A,
   const int                        IA,
   const int                        JA,
   const int                        TAG,
   HPL_T_panel *                    PANEL
)
#else
void HPL_pdpanel_init
( GRID, ALGO, M, N, JB, A, IA, JA, TAG, PANEL )
   HPL_T_grid *                     GRID;
   HPL_T_palg *                     ALGO;
   const int                        M;
   const int                        N;
   const int                        JB;
   HPL_T_pmat *                     A;
   const int                        IA;
   const int                        JA;
   const int                        TAG;
   HPL_T_panel *                    PANEL;
#endif
{
/*
 * Purpose
 * =======
 *
 * HPL_pdpanel_init initializes a panel data structure.
 *
 *
 * Arguments
 * =========
 *
 * GRID    (local input)                 HPL_T_grid *
 *         On entry,  GRID  points  to the data structure containing the
 *         process grid information.
 *
 * ALGO    (global input)                HPL_T_palg *
 *         On entry,  ALGO  points to  the data structure containing the
 *         algorithmic parameters.
 *
 * M       (local input)                 const int
 *         On entry, M specifies the global number of rows of the panel.
 *         M must be at least zero.
 *
 * N       (local input)                 const int
 *         On entry,  N  specifies  the  global number of columns of the
 *         panel and trailing submatrix. N must be at least zero.
 *
 * JB      (global input)                const int
 *         On entry, JB specifies is the number of columns of the panel.
 *         JB must be at least zero.
 *
 * A       (local input/output)          HPL_T_pmat *
 *         On entry, A points to the data structure containing the local
 *         array information.
 *
 * IA      (global input)                const int
 *         On entry,  IA  is  the global row index identifying the panel
 *         and trailing submatrix. IA must be at least zero.
 *
 * JA      (global input)                const int
 *         On entry, JA is the global column index identifying the panel
 *         and trailing submatrix. JA must be at least zero.
 *
 * TAG     (global input)                const int
 *         On entry, TAG is the row broadcast message id.
 *
 * PANEL   (local input/output)          HPL_T_panel *
 *         On entry,  PANEL  points to the data structure containing the
 *         panel information.
 *
 * ---------------------------------------------------------------------
 */
/*
 * .. Local Variables ..
 */
   size_t                     dalign;
   int                        icurcol, icurrow, ii, itmp1, jj, lwork,
                              ml2, mp, mycol, myrow, nb, npcol, nprow,
                              nq, nu;
/* ..
 * .. Executable Statements ..
 */
   PANEL->grid    = GRID;                  /* ptr to the process grid */
   PANEL->algo    = ALGO;               /* ptr to the algo parameters */
   PANEL->pmat    = A;                 /* ptr to the local array info */

   myrow = GRID->myrow; mycol = GRID->mycol;
   nprow = GRID->nprow; npcol = GRID->npcol; nb = A->nb;

   HPL_infog2l( IA, JA, nb, nb, nb, nb, 0, 0, myrow, mycol,
                nprow, npcol, &ii, &jj, &icurrow, &icurcol );
   mp = HPL_numrocI( M, IA, nb, nb, myrow, 0, nprow );
   nq = HPL_numrocI( N, JA, nb, nb, mycol, 0, npcol );
                                         /* ptr to trailing part of A */

   size_t numpinnedbytes = A->ld*JB*sizeof(double);
   if(PANEL->max_pinned_work_size<(size_t)(numpinnedbytes))
   {
      if( PANEL->A  )
      {
         hipHostFree( PANEL->A);
      }
#ifdef VERBOSE_PRINT
      if( ( myrow == 0 ) && ( mycol == 0 ) )
      {printf("Allocating %g GBs of storage on CPU...",((double) numpinnedbytes)/(1024*1024*1024)); fflush(stdout);}
#endif
      hipError_t statusHost = hipHostMalloc(&(PANEL->A), numpinnedbytes,0);
      if( statusHost != hipSuccess) {
         HPL_pabort( __LINE__, "HPL_pdpanel_init",
                     "Panel Host Memory allocation failed" );
         return;
      }
#ifdef VERBOSE_PRINT
   if( ( myrow == 0 ) && ( mycol == 0 ) )
     {printf("done.\n");}
#endif
      PANEL->max_pinned_work_size = (size_t)(numpinnedbytes);
   }

   PANEL->dA      = Mptr( (double *)(A->dA), ii, jj, A->ld );

/*
 * Workspace pointers are initialized to NULL.
 */
   // PANEL->WORK    = NULL;
   PANEL->L2      = NULL;
   PANEL->dL2     = NULL;
   PANEL->L1      = NULL;
   PANEL->dL1     = NULL;
   PANEL->DINFO   = NULL;
   PANEL->U       = NULL;
   PANEL->dU      = NULL;
   // PANEL->IWORK   = NULL;
/*
 * Local lengths, indexes process coordinates
 */
   PANEL->nb      = nb;               /* distribution blocking factor */
   PANEL->jb      = JB;                                /* panel width */
   PANEL->m       = M;      /* global # of rows of trailing part of A */
   PANEL->n       = N;      /* global # of cols of trailing part of A */
   PANEL->ia      = IA;     /* global row index of trailing part of A */
   PANEL->ja      = JA;     /* global col index of trailing part of A */
   PANEL->mp      = mp;      /* local # of rows of trailing part of A */
   PANEL->nq      = nq;      /* local # of cols of trailing part of A */
   PANEL->ii      = ii;      /* local row index of trailing part of A */
   PANEL->jj      = jj;      /* local col index of trailing part of A */
   PANEL->lda     = A->ld;            /* local leading dim of array A */
   PANEL->prow    = icurrow; /* proc row owning 1st row of trailing A */
   PANEL->pcol    = icurcol; /* proc col owning 1st col of trailing A */
   PANEL->msgid   = TAG;     /* message id to be used for panel bcast */
/*
 * Initialize  ldl2 and len to temporary dummy values and Update tag for
 * next panel
 */
   PANEL->ldl2    = 0;               /* local leading dim of array L2 */
   PANEL->len     = 0;           /* length of the buffer to broadcast */
/*
 * Figure out the exact amount of workspace  needed by the factorization
 * and the update - Allocate that space - Finish the panel data structu-
 * re initialization.
 *
 * L1:    JB x JB in all processes
 * DINFO: 1       in all processes
 *
 * We also make an array of necessary intergers for swaps in the update.
 *
 * If nprow is 1, we just allocate an array of 2*JB integers for the swap.
 * When nprow > 1, we allocate the space for the index arrays immediate-
 * ly. The exact size of this array depends on the swapping routine that
 * will be used, so we allocate the maximum:
 *
 *    For HPL_pdlaswp00:
 *       lindxA   is of size at most 2 * JB +
 *       lindxAU  is of size at most 2 * JB
 *
 *    For HPL_pdlaswp01:
 *       lindxA   is of size at most 2 * JB +
 *       lindxAU  is of size at most 2 * JB +
 *       permU    is of size at most 2 * JB
 *
 *       ipiv     is of size at most JB
 *
 * that is  7*JB.
 *
 * We make sure that those three arrays are contiguous in memory for the
 * later panel broadcast (using type punning to put the integer array at
 * the end.  We  also  choose  to put this amount of space right after
 * L2 (when it exist) so that one can receive a contiguous buffer.
 */

   dalign = ALGO->align * sizeof( double );
   size_t lpiv = (6*JB*sizeof(int) + sizeof(double)-1)/(sizeof(double));
   size_t ipivlen = (JB*sizeof(int) + sizeof(double)-1)/(sizeof(double));

   if( npcol == 1 )                             /* P x 1 process grid */
   {                                     /* space for L1, PIV, DINFO */
      PANEL->len = JB * JB + lpiv; //L1, integer arrays
      lwork = ALGO->align + ( PANEL->len + ipivlen + 1);
      if( nprow > 1 )                                 /* space for U */
      { nu = nq - JB; lwork += JB * Mmax( 0, nu ); }

      if(PANEL->max_work_size<(size_t)(lwork) * sizeof( double ))
      {
         if( PANEL->WORK  )
         {
            hipFree( PANEL->dWORK);
            hipHostFree( PANEL->WORK);
         }
         // size_t numbytes = (((size_t)((size_t)(lwork) * sizeof( double )) + (size_t)4095)/(size_t)4096)*(size_t)4096;
         size_t numbytes = (size_t)(lwork) *sizeof( double );

#ifdef VERBOSE_PRINT
         if( ( myrow == 0 ) && ( mycol == 0 ) )
            {printf("Allocating %g GBs of storage on CPU...",((double) numbytes)/(1024*1024*1024)); fflush(stdout);}
#endif
         hipError_t statusHost = hipHostMalloc((void**)&(PANEL->WORK),numbytes, hipHostMallocDefault);

         if(statusHost!=hipSuccess) {
            HPL_pabort( __LINE__, "HPL_pdpanel_init",
                        "Panel Host Memory allocation failed" );
         }

#ifdef VERBOSE_PRINT
         if( ( myrow == 0 ) && ( mycol == 0 ) ){
            printf("done.\n");
            printf("Allocating %g GBs of storage on GPU...",((double) numbytes)/(1024*1024*1024)); fflush(stdout);
         }
#endif
         hipError_t statusDevice = hipMalloc((void**)&(PANEL->dWORK),numbytes);

         if(statusDevice!=hipSuccess) {
            HPL_pabort( __LINE__, "HPL_pdpanel_init",
                        "Panel Device Memory allocation failed" );
         }
#ifdef VERBOSE_PRINT
         if( ( myrow == 0 ) && ( mycol == 0 ) )
            printf("done.\n");
#endif
         PANEL->max_work_size = (size_t)(lwork) * sizeof( double );
      }
/*
 * Initialize the pointers of the panel structure  -  Always re-use A in
 * the only process column
 */
      PANEL->ldl2  = A->ld;
      PANEL->dL2   = PANEL->dA + ( myrow == icurrow ? JB : 0 );
      PANEL->L2    = PANEL->A  + ( myrow == icurrow ? JB : 0 );
      PANEL->dL1   = (double *)HPL_PTR( PANEL->dWORK, dalign );
      PANEL->L1    = (double *)HPL_PTR( PANEL->WORK, dalign );

      PANEL->dlindxA = (int *) (PANEL->dL1 + JB * JB);
      PANEL->lindxA  = (int *) (PANEL->L1 + JB * JB);
      PANEL->dlindxAU = PANEL->dlindxA  + 2*JB;
      PANEL->lindxAU  = PANEL->lindxA   + 2*JB;
      PANEL->dpermU   = PANEL->dlindxAU + 2*JB;
      PANEL->permU    = PANEL->lindxAU  + 2*JB;

      //Put ipiv array at the end
      PANEL->dipiv = PANEL->dpermU + JB;
      PANEL->ipiv  = PANEL->permU  + JB;

      PANEL->DINFO = ((double*) PANEL->lindxA)  + lpiv + ipivlen;
      PANEL->dDINFO= ((double*) PANEL->dlindxA) + lpiv + ipivlen;

      *(PANEL->DINFO) = 0.0;
      PANEL->U     = ( nprow > 1 ? PANEL->DINFO + 1: NULL );
      PANEL->dU    = ( nprow > 1 ? PANEL->dDINFO+ 1: NULL );
   }
   else
   {                                        /* space for L2, L1, DPIV */
      ml2 = ( myrow == icurrow ? mp - JB : mp ); ml2 = Mmax( 0, ml2 );

      itmp1 = JB*JB + lpiv + ipivlen;  //L1, integer arrays
      PANEL->len = ml2*JB + itmp1;

#ifdef HPL_COPY_L
      lwork = ALGO->align + PANEL->len  + 1;
#else
      lwork = ALGO->align + ( mycol == icurcol ? itmp1 : PANEL->len ) + 1;
#endif
      if( nprow > 1 )                                 /* space for U */
      {
         nu = ( mycol == icurcol ? nq - JB : nq );
         lwork += JB * Mmax( 0, nu );
      }

      if(PANEL->max_work_size<(size_t)(lwork) * sizeof( double ))
      {
        if( PANEL->WORK  )
        {
          hipFree( PANEL->dWORK);
          hipHostFree( PANEL->WORK);
        }
        // size_t numbytes = (((size_t)((size_t)(lwork) * sizeof( double )) + (size_t)4095)/(size_t)4096)*(size_t)4096;
        size_t numbytes = (size_t)(lwork) *sizeof( double );

#ifdef VERBOSE_PRINT
         if( ( myrow == 0 ) && ( mycol == 0 ) )
            {printf("Allocating %g GBs of storage on CPU...",((double) numbytes)/(1024*1024*1024)); fflush(stdout);}
#endif
         hipError_t statusHost = hipHostMalloc((void**)&(PANEL->WORK),numbytes, hipHostMallocDefault);

         if(statusHost!=hipSuccess) {
            HPL_pabort( __LINE__, "HPL_pdpanel_init",
                        "Panel Host Memory allocation failed" );
         }

#ifdef VERBOSE_PRINT
         if( ( myrow == 0 ) && ( mycol == 0 ) ){
            printf("done.\n");
            printf("Allocating %g GBs of storage on GPU...",((double) numbytes)/(1024*1024*1024)); fflush(stdout);
         }
#endif
         hipError_t statusDevice = hipMalloc((void**)&(PANEL->dWORK),numbytes);

         if(statusDevice!=hipSuccess) {
            HPL_pabort( __LINE__, "HPL_pdpanel_init",
                        "Panel Device Memory allocation failed" );
         }
#ifdef VERBOSE_PRINT
         if( ( myrow == 0 ) && ( mycol == 0 ) )
            printf("done.\n");
#endif
         PANEL->max_work_size = (size_t)(lwork) * sizeof( double );
      }
/*
 * Initialize the pointers of the panel structure - Re-use A in the cur-
 * rent process column when HPL_COPY_L is not defined.
 */
#ifdef HPL_COPY_L
      PANEL->dL2   = (double *)HPL_PTR( PANEL->dWORK, dalign );
      PANEL->dL1   = PANEL->dL2 + ml2 * JB;
      PANEL->L2    = (double *)HPL_PTR( PANEL->WORK, dalign );
      PANEL->L1    = PANEL->L2 + ml2 * JB;
      PANEL->ldl2  = Mmax( 1, ml2 );
#else
      if( mycol == icurcol )
      {
         PANEL->L2   = PANEL->A + ( myrow == icurrow ? JB : 0 );
         PANEL->dL2  = PANEL->dA + ( myrow == icurrow ? JB : 0 );
         PANEL->ldl2 = A->ld;
         PANEL->L1   = (double *)HPL_PTR( PANEL->WORK, dalign );
         PANEL->dL1   = (double *)HPL_PTR( PANEL->dWORK, dalign );
      }
      else
      {
         PANEL->dL2   = (double *)HPL_PTR( PANEL->dWORK, dalign );
         PANEL->dL1   = PANEL->dL2 + ml2 * JB;

         PANEL->L2   = (double *)HPL_PTR( PANEL->WORK, dalign );
         PANEL->L1   = PANEL->L2 + ml2 * JB;
         PANEL->ldl2 = Mmax( 1, ml2 );
      }
#endif

      PANEL->dlindxA = (int *) (PANEL->dL1 + JB * JB);
      PANEL->lindxA  = (int *) (PANEL->L1  + JB * JB);
      PANEL->dlindxAU = PANEL->dlindxA  + 2*JB;
      PANEL->lindxAU  = PANEL->lindxA   + 2*JB;
      PANEL->dpermU   = PANEL->dlindxAU + 2*JB;
      PANEL->permU    = PANEL->lindxAU  + 2*JB;

      PANEL->dipiv = PANEL->dpermU + JB;
      PANEL->ipiv  = PANEL->permU  + JB;

      PANEL->DINFO = ((double*) PANEL->lindxA)  + lpiv + ipivlen;
      PANEL->dDINFO= ((double*) PANEL->dlindxA) + lpiv + ipivlen;

      *(PANEL->DINFO) = 0.0;
      PANEL->U     = ( nprow > 1 ? PANEL->DINFO + 1 : NULL );
      PANEL->dU    = ( nprow > 1 ? PANEL->dDINFO + 1: NULL );
   }
/*
 * If nprow is 1, we just allocate an array of JB integers to store the
 * pivot IDs during factoring, and a scratch array of mp integers.
 * When nprow > 1, we allocate the space for the index arrays immediate-
 * ly. The exact size of this array depends on the swapping routine that
 * will be used, so we allocate the maximum:
 *
 *    IWORK[0] is of size at most 1      +
 *    IPL      is of size at most 1      +
 *    IPID     is of size at most 4 * JB +
 *    IPIV     is of size at most JB     +
 *    SCRATCH  is of size at most MP
 *
 *    For HPL_pdlaswp00:
 *       llen     is of size at most NPROW  +
 *       llen_sv  is of size at most NPROW.
 *
 *    For HPL_pdlaswp01:
 *       ipA      is of size at most 1      +
 *       iplen    is of size at most NPROW  + 1 +
 *       ipmap    is of size at most NPROW  +
 *       ipmapm1  is of size at most NPROW  +
 *       iwork    is of size at most MAX( 2*JB, NPROW+1 ).
 *
 * that is  mp + 3 + 5*JB + MAX(2*NPROW, 3*NPROW+1+MAX(2*JB,NPROW+1))
 *       =  mp + 4 + 5*JB + 3*NPROW + MAX( 2*JB, NPROW+1 ).
 *
 * We use the fist entry of this to work array  to indicate  whether the
 * the  local  index arrays have already been computed,  and if yes,  by
 * which function:
 *    IWORK[0] = -1: no index arrays have been computed so far;
 *    IWORK[0] =  0: HPL_pdlaswp00 already computed those arrays;
 *    IWORK[0] =  1: HPL_pdlaswp01 already computed those arrays;
 * This allows to save some redundant and useless computations.
 */
   if( nprow == 1 ) { lwork = mp + JB; }
   else
   {
      itmp1 = (JB << 1); lwork = nprow + 1; itmp1 = Mmax( itmp1, lwork );
      lwork = mp + 4 + (5 * JB) + (3 * nprow) + itmp1;
   }

   if(PANEL->max_iwork_size<(size_t)(lwork) * sizeof( int ))
   {
      if( PANEL->IWORK  )
      {
        hipHostFree( PANEL->IWORK);
      }
      size_t numbytes = (size_t)(lwork) *sizeof( int );

      hipError_t statusHost = hipHostMalloc((void**)&(PANEL->IWORK),numbytes, hipHostMallocDefault);
      if(statusHost!=hipSuccess) {
         HPL_pabort( __LINE__, "HPL_pdpanel_init",
                     "Panel Host Integer Memory allocation failed" );
      }
      PANEL->max_iwork_size = (size_t)(lwork) * sizeof( int );
   }

   if (lwork)
    *(PANEL->IWORK) = -1;
/*
 * End of HPL_pdpanel_init
 */
}
