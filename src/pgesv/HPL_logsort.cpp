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

void HPL_logsort(const int NPROCS,
                 const int ICURROC,
                 int*      IPLEN,
                 int*      IPMAP,
                 int*      IPMAPM1) {
  /*
   * Purpose
   * =======
   *
   * HPL_logsort computes an array  IPMAP  and  its inverse  IPMAPM1  that
   * contain  the logarithmic sorted processes id with repect to the local
   * number of rows of  U  that they own. This is necessary to ensure that
   * the logarithmic spreading of U is optimal in terms of number of steps
   * and communication volume as well.  In other words,  the larget pieces
   * of U will be sent a minimal number of times.
   *
   * Arguments
   * =========
   *
   * NPROCS  (global input)                const int
   *         On entry, NPROCS  specifies the number of process rows in the
   *         process grid. NPROCS is at least one.
   *
   * ICURROC (global input)                const int
   *         On entry, ICURROC is the source process row.
   *
   * IPLEN   (global input/output)         int *
   *         On entry, IPLEN is an array of dimension NPROCS+1,  such that
   *         IPLEN[0] is 0, and IPLEN[i] contains the number of rows of U,
   *         that process i-1 has.  On exit,  IPLEN[i]  is  the number  of
   *         rows of U  in the processes before process IPMAP[i] after the
   *         sort,  with  the convention that  IPLEN[NPROCS] is  the total
   *         number  of rows  of the panel.  In other words,  IPLEN[i+1] -
   *         IPLEN[i] is  the  number of rows of A that should be moved to
   *         the process IPMAP[i].  IPLEN  is such that the number of rows
   *         of  the  source process  row is IPLEN[1] - IPLEN[0],  and the
   *         remaining  entries  of  this  array  are  sorted  so that the
   *         quantities IPLEN[i+1]-IPLEN[i] are logarithmically sorted.
   *
   * IPMAP   (global output)               int *
   *         On entry,  IPMAP  is an array of dimension  NPROCS.  On exit,
   *         array contains  the logarithmic mapping of the processes.  In
   *         other words, IPMAP[myroc] is the corresponding sorted process
   *         coordinate.
   *
   * IPMAPM1 (global output)               int *
   *         On entry, IPMAPM1  is an array of dimension NPROCS.  On exit,
   *         this  array  contains  the inverse of the logarithmic mapping
   *         contained  in  IPMAP:  IPMAPM1[ IPMAP[i] ] = i,  for all i in
   *         [0.. NPROCS)
   *
   * ---------------------------------------------------------------------
   */

  int dist, i, ip, iplen_i, iplen_j, itmp, j, k;

  /*
   * Compute the  logarithmic distance between process j and process 0, as
   * well as the maximum logarithmic distance. IPMAPM1 is workarray here.
   */
  for(j = 0, dist = 0; j < NPROCS; j++) {
    IPMAP[j] = MModAdd(j, ICURROC, NPROCS);
    ip       = j;
    itmp     = 0;
    do {
      if(ip & 1) itmp++;
      ip >>= 1;
    } while(ip);
    IPMAPM1[j] = itmp;
    if(itmp > dist) dist = itmp;
  }
  /*
   * Shift IPLEN[1..NPROCS]  of ICURROC places,  so that  IPLEN[1]  is now
   * what used to be IPLEN[ICURROC+1]. Initialize IPMAP,  so that IPMAP[0]
   * is ICURROC.
   */
  for(j = 0; j < ICURROC; j++) {
    for(i = 2, itmp = IPLEN[1]; i <= NPROCS; i++) IPLEN[i - 1] = IPLEN[i];
    IPLEN[NPROCS] = itmp;
  }
  /*
   * logarithmic sort
   */
  for(k = 1; k <= dist; k++) {
    for(j = 1; j < NPROCS; j++) {
      if(IPMAPM1[j] == k) {
        for(i = 2; i < NPROCS; i++) {
          if(k < IPMAPM1[i]) {
            iplen_i = IPLEN[i + 1];
            iplen_j = IPLEN[j + 1];

            if(iplen_j < iplen_i) {
              IPLEN[j + 1] = iplen_i;
              IPLEN[i + 1] = iplen_j;
              itmp         = IPMAP[j];
              IPMAP[j]     = IPMAP[i];
              IPMAP[i]     = itmp;
            }
          }
        }
      }
    }
  }
  /*
   * Compute IPLEN and IPMAPM1 (the inverse of IPMAP)
   */
  IPLEN[0] = 0;

  for(i = 0; i < NPROCS; i++) {
    IPMAPM1[IPMAP[i]] = i;
    IPLEN[i + 1] += IPLEN[i];
  }
}
