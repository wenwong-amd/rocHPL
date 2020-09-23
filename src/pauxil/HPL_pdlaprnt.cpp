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

#include "hpl.hpp"

void HPL_pdlaprnt(const HPL_T_grid* GRID,
                  const int         M,
                  const int         N,
                  const int         NB,
                  double*           A,
                  const int         LDA,
                  const int         IAROW,
                  const int         IACOL,
                  const char*       CMATNM) {
  /*
   * Purpose
   * =======
   *
   * HPL_pdlaprnt prints  to  standard  error a distributed matrix A. The
   * local pieces of  A  are sent to the process of coordinates  (0,0)  in
   * the grid and then printed.
   *
   * Arguments
   * =========
   *
   * GRID    (local input)                 const HPL_T_grid *
   *         On entry,  GRID  points  to the data structure containing the
   *         process grid information.
   *
   * M       (global input)                const int
   *         On entry,  M  specifies the number of rows of the coefficient
   *         matrix A. M must be at least zero.
   *
   * N       (global input)                const int
   *         On  entry,   N   specifies  the  number  of  columns  of  the
   *         coefficient matrix A. N must be at least zero.
   *
   * NB      (global input)                const int
   *         On entry,  NB specifies the blocking factor used to partition
   *         and distribute the matrix. NB must be larger than one.
   *
   * A       (local input)                 double *
   *         On entry,  A  points to an  array of dimension (LDA,LocQ(N)).
   *         This array contains the coefficient matrix to be printed.
   *
   * LDA     (local input)                 const int
   *         On entry, LDA specifies the leading dimension of the array A.
   *         LDA must be at least max(1,LocP(M)).
   *
   * IAROW   (global input)                const int
   *         On entry,  IAROW  specifies the row process coordinate owning
   *         the  first row of A.  IAROW  must be  larger than or equal to
   *         zero and less than NPROW.
   *
   * IACOL   (global input)                const int
   *         On entry,  IACOL  specifies  the  column  process  coordinate
   *         owning the  first column  of A. IACOL  must be larger than or
   *         equal to zero and less than NPCOL.
   *
   * CMATNM  (global input)                const char *
   *         On entry, CMATNM is the name of the matrix to be printed.
   *
   * ---------------------------------------------------------------------
   */

  MPI_Comm Acomm;
  double*  buf = NULL;
  int h, i, ib, icurcol = IACOL, icurrow = IAROW, ii = 0, j, jb, jj = 0, mycol,
                myrow, npcol, nprow, src;

  (void)HPL_grid_info(GRID, &nprow, &npcol, &myrow, &mycol);
  Acomm = GRID->all_comm;
  if((myrow == 0) && (mycol == 0))
    buf = (double*)malloc((size_t)(NB) * sizeof(double));

  for(j = 0; j < N; j += NB) {
    jb = N - j;
    jb = Mmin(jb, NB);
    for(h = 0; h < jb; h++) {
      (void)HPL_barrier(Acomm);

      for(i = 0; i < M; i += NB) {
        ib = M - i;
        ib = Mmin(ib, NB);
        if((icurrow == 0) && (icurcol == 0)) {
          if((myrow == 0) && (mycol == 0))
            HPL_dlaprnt(
                ib, 1, Mptr(A, ii, jj + h, LDA), i + 1, j + h + 1, LDA, CMATNM);
        } else {
          if((myrow == icurrow) && (mycol == icurcol)) {
            (void)HPL_send(
                Mptr(A, ii, jj + h, LDA), ib, 0, 9000 + (j + h) * M + i, Acomm);
          } else if((myrow == 0) && (mycol == 0)) {
            src = HPL_pnum(GRID, icurrow, icurcol);
            (void)HPL_recv(buf, ib, src, 9000 + (j + h) * M + i, Acomm);
            HPL_dlaprnt(ib, 1, buf, i + 1, j + h + 1, NB, CMATNM);
          }
        }
        if(myrow == icurrow) ii += ib;
        icurrow = MModAdd1(icurrow, nprow);
        (void)HPL_barrier(Acomm);
      }
      ii      = 0;
      icurrow = IAROW;
    }
    if(mycol == icurcol) jj += jb;
    icurcol = MModAdd1(icurcol, npcol);
    (void)HPL_barrier(Acomm);
  }
  if((myrow == 0) && (mycol == 0) && (buf)) free(buf);
}
