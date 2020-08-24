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

#include "hpl.h"
#include "rocrand.h"

#define BLOCK_SIZE 512

__global__ void hpl_init_shift(double* __restrict__ A, const size_t n) {

  const size_t id = threadIdx.x + ((size_t)blockIdx.x) * BLOCK_SIZE;

  if(id < n) A[id] -= 0.5;
}

void HPL_pdmatgen(const HPL_T_grid* GRID,
                  const int         M,
                  const int         N,
                  const int         NB,
                  double*           A,
                  const int         LDA,
                  const int         ISEED) {
  /*
   * Purpose
   * =======
   *
   * HPL_pdmatgen generates (or regenerates) a parallel random matrix A.
   *
   * The  pseudo-random  generator uses the linear congruential algorithm:
   * X(n+1) = (a * X(n) + c) mod m  as  described  in the  Art of Computer
   * Programming, Knuth 1973, Vol. 2.
   *
   * Arguments
   * =========
   *
   * GRID    (local input)                 const HPL_T_grid *
   *         On entry,  GRID  points  to the data structure containing the
   *         process grid information.
   *
   * M       (global input)                const int
   *         On entry,  M  specifies  the number  of rows of the matrix A.
   *         M must be at least zero.
   *
   * N       (global input)                const int
   *         On entry,  N specifies the number of columns of the matrix A.
   *         N must be at least zero.
   *
   * NB      (global input)                const int
   *         On entry,  NB specifies the blocking factor used to partition
   *         and distribute the matrix A. NB must be larger than one.
   *
   * A       (local output)                double *
   *         On entry,  A  points  to an array of dimension (LDA,LocQ(N)).
   *         On exit, this array contains the coefficients of the randomly
   *         generated matrix.
   *
   * LDA     (local input)                 const int
   *         On entry, LDA specifies the leading dimension of the array A.
   *         LDA must be at least max(1,LocP(M)).
   *
   * ISEED   (global input)                const int
   *         On entry, ISEED  specifies  the  seed  number to generate the
   *         matrix A. ISEED must be at least zero.
   *
   * ---------------------------------------------------------------------
   */

  int iadd[2], ia1[2], ia2[2], ia3[2], ia4[2], ia5[2], ib1[2], ib2[2], ib3[2],
      ic1[2], ic2[2], ic3[2], ic4[2], ic5[2], iran1[2], iran2[2], iran3[2],
      iran4[2], itmp1[2], itmp2[2], itmp3[2], jseed[2], mult[2];
  int ib, iblk, ik, jb, jblk, jk, jump1, jump2, jump3, jump4, jump5, jump6,
      jump7, lmb, lnb, mblks, mp, mycol, myrow, nblks, npcol, nprow, nq;

  (void)HPL_grid_info(GRID, &nprow, &npcol, &myrow, &mycol);

  mult[0]  = HPL_MULT0;
  mult[1]  = HPL_MULT1;
  iadd[0]  = HPL_IADD0;
  iadd[1]  = HPL_IADD1;
  jseed[0] = ISEED;
  jseed[1] = 0;
  /*
   * Generate an M by N matrix starting in process (0,0)
   */
  Mnumroc(mp, M, NB, NB, myrow, 0, nprow);
  Mnumroc(nq, N, NB, NB, mycol, 0, npcol);

  if((mp <= 0) || (nq <= 0)) return;
  /*
   * Local number of blocks and size of the last one
   */
  mblks = (mp + NB - 1) / NB;
  lmb   = mp - ((mp - 1) / NB) * NB;
  nblks = (nq + NB - 1) / NB;
  lnb   = nq - ((nq - 1) / NB) * NB;
  /*
   * Compute multiplier/adder for various jumps in random sequence
   */
  jump1 = 1;
  jump2 = nprow * NB;
  jump3 = M;
  jump4 = npcol * NB;
  jump5 = NB;
  jump6 = mycol;
  jump7 = myrow * NB;

  HPL_xjumpm(jump1, mult, iadd, jseed, iran1, ia1, ic1);
  HPL_xjumpm(jump2, mult, iadd, iran1, itmp1, ia2, ic2);
  HPL_xjumpm(jump3, mult, iadd, iran1, itmp1, ia3, ic3);
  HPL_xjumpm(jump4, ia3, ic3, iran1, itmp1, ia4, ic4);
  HPL_xjumpm(jump5, ia3, ic3, iran1, itmp1, ia5, ic5);
  HPL_xjumpm(jump6, ia5, ic5, iran1, itmp3, itmp1, itmp2);
  HPL_xjumpm(jump7, mult, iadd, itmp3, iran1, itmp1, itmp2);
  HPL_setran(0, iran1);
  HPL_setran(1, ia1);
  HPL_setran(2, ic1);
  /*
   * Save value of first number in sequence
   */
  ib1[0] = iran1[0];
  ib1[1] = iran1[1];
  ib2[0] = iran1[0];
  ib2[1] = iran1[1];
  ib3[0] = iran1[0];
  ib3[1] = iran1[1];

#if 1
  /* Initialize on GPU */
  mp                      = (mp < LDA) ? LDA : mp;
  unsigned long long pos1 = myrow * nq + mycol * mp * M;

  rocrand_generator generator;
  rocrand_create_generator(&generator, ROCRAND_RNG_PSEUDO_DEFAULT);
  rocrand_set_seed(generator, ISEED);
  rocrand_set_offset(generator, pos1);

  // generate values between [0,1]
  rocrand_generate_uniform_double(generator, A, ((size_t)mp) * nq);
  hipDeviceSynchronize();

  // shift values to [-0.5,0.5]
  size_t grid_size = (((size_t)mp) * nq + BLOCK_SIZE - 1) / BLOCK_SIZE;
  hipLaunchKernelGGL((hpl_init_shift),
                     dim3(grid_size),
                     dim3(BLOCK_SIZE),
                     0,
                     0,
                     A,
                     ((size_t)mp) * nq);

  hipDeviceSynchronize();

  rocrand_destroy_generator(generator);

#else // original initialization

  for(jblk = 0; jblk < nblks; jblk++) {
    jb = (jblk == nblks - 1 ? lnb : NB);
    for(jk = 0; jk < jb; jk++) {
      for(iblk = 0; iblk < mblks; iblk++) {
        ib = (iblk == mblks - 1 ? lmb : NB);
        for(ik = 0; ik < ib; A++, ik++) *A = HPL_rand();
        HPL_jumpit(ia2, ic2, ib1, iran2);
        ib1[0] = iran2[0];
        ib1[1] = iran2[1];
      }
      A += LDA - mp;
      HPL_jumpit(ia3, ic3, ib2, iran3);
      ib1[0] = iran3[0];
      ib1[1] = iran3[1];
      ib2[0] = iran3[0];
      ib2[1] = iran3[1];
    }
    HPL_jumpit(ia4, ic4, ib3, iran4);
    ib1[0] = iran4[0];
    ib1[1] = iran4[1];
    ib2[0] = iran4[0];
    ib2[1] = iran4[1];
    ib3[0] = iran4[0];
    ib3[1] = iran4[1];
  }
#endif
}
