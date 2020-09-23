/* ---------------------------------------------------------------------
 * -- High Performance Computing Linpack Benchmark (HPL)
 *    Noel Chalmers
 *    (C) 2018-2020 Advanced Micro Devices, Inc.
 *    See the rocHPL/LICENCE file for details.
 *
 *    SPDX-License-Identifier: (BSD-3-Clause)
 * ---------------------------------------------------------------------
 */
#include "hpl.hpp"

void HPL_unroll_ipiv(const int mp,
                     const int jb,
                     int*      ipiv,
                     int*      ipiv_ex,
                     int*      upiv) {

  for(int i = 0; i < mp; i++) { upiv[i] = i; } // initialize ids
  for(int i = 0; i < jb; i++) {                // swap ids
    int id        = upiv[i];
    upiv[i]       = upiv[ipiv[i]];
    upiv[ipiv[i]] = id;
  }

  for(int i = 0; i < jb; i++) { ipiv_ex[i] = -1; }

  int cnt = 0;
  for(int i = jb; i < mp; i++) { // find swapped ids outside of panel
    if(upiv[i] < jb) { ipiv_ex[upiv[i]] = i; }
  }
}

void HPL_pdpanel_SendToDevice(HPL_T_panel* PANEL) {
  double *   A, *dA;
  int        jb, i, ml2;
  static int equil = -1;

  jb = PANEL->jb;

  if(jb <= 0) return;

#ifdef GPU_AWARE_MPI
  // only the root column copies to device
  if(PANEL->grid->mycol == PANEL->pcol) {
#endif

    if(PANEL->grid->nprow == 1) {

      // unroll pivoting and send to device now
      int* ipiv    = PANEL->ipiv;
      int* ipiv_ex = PANEL->ipiv + jb;
      int* upiv    = PANEL->IWORK + jb; // scratch space

      for(i = 0; i < jb; i++) { ipiv[i] -= PANEL->ii; } // shift
      HPL_unroll_ipiv(PANEL->mp, jb, ipiv, ipiv_ex, upiv);

      int* dipiv    = PANEL->dipiv;
      int* dipiv_ex = PANEL->dipiv + jb;

      hipMemcpy2DAsync(dipiv,
                       jb * sizeof(int),
                       upiv,
                       jb * sizeof(int),
                       jb * sizeof(int),
                       1,
                       hipMemcpyHostToDevice,
                       dataStream);
      hipMemcpy2DAsync(dipiv_ex,
                       jb * sizeof(int),
                       ipiv_ex,
                       jb * sizeof(int),
                       jb * sizeof(int),
                       1,
                       hipMemcpyHostToDevice,
                       dataStream);

    } else {

      if(equil == -1) equil = PANEL->algo->equil;

      int  k       = (int)((unsigned int)(jb) << 1);
      int* iflag   = PANEL->IWORK;
      int* ipl     = iflag + 1;
      int* ipID    = ipl + 1;
      int* ipA     = ipID + ((unsigned int)(k) << 1);
      int* iplen   = ipA + 1;
      int* ipmap   = iplen + PANEL->grid->nprow + 1;
      int* ipmapm1 = ipmap + PANEL->grid->nprow;
      int* upiv    = ipmapm1 + PANEL->grid->nprow;
      int* iwork   = upiv + PANEL->mp;

      int* lindxA   = PANEL->lindxA;
      int* lindxAU  = PANEL->lindxAU;
      int* permU    = PANEL->permU;
      int* permU_ex = permU + jb;

      int* dlindxA   = PANEL->dlindxA;
      int* dlindxAU  = PANEL->dlindxAU;
      int* dpermU    = PANEL->dpermU;
      int* dpermU_ex = dpermU + jb;

      if(*iflag == -1) /* no index arrays have been computed so far */
      {
        HPL_pipid(PANEL, ipl, ipID);
        HPL_plindx1(PANEL,
                    *ipl,
                    ipID,
                    ipA,
                    lindxA,
                    lindxAU,
                    iplen,
                    ipmap,
                    ipmapm1,
                    permU,
                    iwork);
        *iflag = 1;
      } else if(*iflag == 0) /* HPL_pdlaswp00N called before: reuse ipID */
      {
        HPL_plindx1(PANEL,
                    *ipl,
                    ipID,
                    ipA,
                    lindxA,
                    lindxAU,
                    iplen,
                    ipmap,
                    ipmapm1,
                    permU,
                    iwork);
        *iflag = 1;
      } else if((*iflag == 1) &&
                (equil != 0)) { /* HPL_pdlaswp01N was call before only
                                   re-compute IPLEN, IPMAP */
        HPL_plindx10(PANEL, *ipl, ipID, iplen, ipmap, ipmapm1);
        *iflag = 1;
      }

      int N = Mmax(*ipA, jb);
      if(N > 0) {
        hipMemcpy2DAsync(dlindxA,
                         k * sizeof(int),
                         lindxA,
                         k * sizeof(int),
                         N * sizeof(int),
                         1,
                         hipMemcpyHostToDevice,
                         dataStream);
        hipMemcpy2DAsync(dlindxAU,
                         k * sizeof(int),
                         lindxAU,
                         k * sizeof(int),
                         N * sizeof(int),
                         1,
                         hipMemcpyHostToDevice,
                         dataStream);
      }

      if((PANEL->algo->upfun == HPL_pdupdateNN) ||
         (PANEL->algo->upfun == HPL_pdupdateTN)) {
        HPL_unroll_ipiv(jb, jb, permU, permU_ex, upiv);
        hipMemcpy2DAsync(dpermU_ex,
                         jb * sizeof(int),
                         permU_ex,
                         jb * sizeof(int),
                         jb * sizeof(int),
                         1,
                         hipMemcpyHostToDevice,
                         dataStream);
        hipMemcpy2DAsync(dpermU,
                         jb * sizeof(int),
                         upiv,
                         jb * sizeof(int),
                         jb * sizeof(int),
                         1,
                         hipMemcpyHostToDevice,
                         dataStream);
      } else {
        hipMemcpy2DAsync(dpermU,
                         jb * sizeof(int),
                         permU,
                         jb * sizeof(int),
                         jb * sizeof(int),
                         1,
                         hipMemcpyHostToDevice,
                         dataStream);
      }
    }

#ifdef GPU_AWARE_MPI
  }
#endif

  // copy A and/or L2
  if(PANEL->grid->mycol == PANEL->pcol) {
    // A  = Mptr( PANEL->A,  0, -jb, PANEL->lda );
    A  = Mptr(PANEL->A, 0, 0, PANEL->lda);
    dA = Mptr(PANEL->dA, 0, -jb, PANEL->dlda);

    if(PANEL->mp > 0)
      hipMemcpy2DAsync(dA,
                       PANEL->dlda * sizeof(double),
                       A,
                       PANEL->lda * sizeof(double),
                       PANEL->mp * sizeof(double),
                       jb,
                       hipMemcpyHostToDevice,
                       dataStream);

    if(PANEL->grid->npcol > 1) { // L2 is its own array
      if(PANEL->grid->myrow == PANEL->prow) {
        if((PANEL->mp - jb) > 0)
          hipMemcpy2DAsync(PANEL->dL2,
                           PANEL->dldl2 * sizeof(double),
                           Mptr(PANEL->dA, jb, -jb, PANEL->dlda),
                           PANEL->dlda * sizeof(double),
                           (PANEL->mp - jb) * sizeof(double),
                           jb,
                           hipMemcpyDeviceToDevice,
                           dataStream);
      } else {
        if((PANEL->mp) > 0)
          hipMemcpy2DAsync(PANEL->dL2,
                           PANEL->dldl2 * sizeof(double),
                           Mptr(PANEL->dA, 0, -jb, PANEL->dlda),
                           PANEL->dlda * sizeof(double),
                           (PANEL->mp) * sizeof(double),
                           jb,
                           hipMemcpyDeviceToDevice,
                           dataStream);
      }
    }

    // copy L1
    hipMemcpy2DAsync(PANEL->dL1,
                     jb * sizeof(double),
                     PANEL->L1,
                     jb * sizeof(double),
                     jb * sizeof(double),
                     jb,
                     hipMemcpyHostToDevice,
                     dataStream);

  } else {

#if !defined(GPU_AWARE_MPI)
    // L2+L1 were recieved via MPI, send them to device
    ml2 = (PANEL->grid->myrow == PANEL->prow ? PANEL->mp - jb : PANEL->mp);
    if(ml2 > 0)
      hipMemcpy2DAsync(PANEL->dL2,
                       PANEL->dldl2 * sizeof(double),
                       PANEL->L2,
                       PANEL->ldl2 * sizeof(double),
                       ml2 * sizeof(double),
                       jb,
                       hipMemcpyHostToDevice,
                       dataStream);

    // copy L1
    hipMemcpy2DAsync(PANEL->dL1,
                     jb * sizeof(double),
                     PANEL->L1,
                     jb * sizeof(double),
                     jb * sizeof(double),
                     jb,
                     hipMemcpyHostToDevice,
                     dataStream);
#endif
  }
}
