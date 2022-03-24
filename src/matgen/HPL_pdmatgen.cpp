/* ---------------------------------------------------------------------
 * -- High Performance Computing Linpack Benchmark (HPL)
 *    Noel Chalmers
 *    (C) 2018-2021 Advanced Micro Devices, Inc.
 *    See the rocHPL/LICENCE file for details.
 *
 *    SPDX-License-Identifier: (BSD-3-Clause)
 * ---------------------------------------------------------------------
 */

#include "hpl.hpp"
#include <hip/hip_runtime_api.h>
#include <cassert>
#include <unistd.h>

const int max_nthreads = 128;

int HPL_pdmatgen(HPL_T_test* TEST,
                 HPL_T_grid* GRID,
                 HPL_T_palg* ALGO,
                 HPL_T_pmat* mat,
                 const int   N,
                 const int   NB) {

  int ii, ip2, im4096;
  int mycol, myrow, npcol, nprow, nq, info[3];
  (void)HPL_grid_info(GRID, &nprow, &npcol, &myrow, &mycol);

  mat->n    = N;
  mat->nb   = NB;
  mat->info = 0;
  mat->mp   = HPL_numroc(N, NB, NB, myrow, 0, nprow);
  nq        = HPL_numroc(N, NB, NB, mycol, 0, npcol);
  /*
   * Allocate matrix, right-hand-side, and vector solution x. [ A | b ] is
   * N by N+1.  One column is added in every process column for the solve.
   * The  result  however  is stored in a 1 x N vector replicated in every
   * process row. In every process, A is lda * (nq+1), x is 1 * nq and the
   * workspace is mp.
   *
   * Ensure that lda is a multiple of ALIGN and not a power of 2, and not
   * a multiple of 4096 bytes
   */
  mat->ld = ((Mmax(1, mat->mp) - 1) / ALGO->align) * ALGO->align;
  do {
    ii  = (mat->ld += ALGO->align);
    ip2 = 1;
    while(ii > 1) {
      ii >>= 1;
      ip2 <<= 1;
    }
    im4096 = (mat->ld % 512 ) ? 0 : 1;
  } while((mat->ld == ip2) || im4096);

  mat->nq = nq + 1;

  mat->dA = nullptr;
  mat->dX = nullptr;

  mat->dW = nullptr;
  mat->W  = nullptr;

  /* Create a rocBLAS handle */
  rocblas_create_handle(&handle);
  rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  rocblas_initialize();
  rocblas_set_stream(handle, computeStream);

  /*
   * Allocate dynamic memory
   */

  // allocate on device
  size_t numbytes = ((size_t)(mat->ld) * (size_t)(mat->nq)) * sizeof(double);

#ifdef HPL_VERBOSE_PRINT
  if((myrow == 0) && (mycol == 0)) {
    printf("Allocating %g GBs of storage on GPU...",
           ((double)numbytes) / (1024 * 1024 * 1024));
    fflush(stdout);
  }
#endif

  hipMalloc(&(mat->dA), numbytes);

  /*Check matrix allocation is valid*/
#ifdef HPL_VERBOSE_PRINT
  if (mat->dA==NULL) {
    char host_name[MPI_MAX_PROCESSOR_NAME];
    int rank, namelen;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Get_processor_name(host_name, &namelen);

    printf("Matrix allocation on node %s, rank %d, failed. \n",
           host_name,
           rank);
  }
#endif
  info[0] = (mat->dA == NULL);
  info[1] = myrow;
  info[2] = mycol;
  (void)HPL_all_reduce((void*)(info), 3, HPL_INT, HPL_MAX, GRID->all_comm);
  if(info[0] != 0) {
    HPL_pwarn(TEST->outfp,
              __LINE__,
              "HPL_pdmatgen",
              "[%d,%d] %s",
              info[1],
              info[2],
              "Device memory allocation failed for A and b. Skip.");
    return HPL_FAILURE;
  }
#ifdef HPL_VERBOSE_PRINT
  if((myrow == 0) && (mycol == 0)) printf("done.\n");
#endif

  // seperate space for X vector
  hipMalloc(&(mat->dX), mat->nq * sizeof(double));

  /*Check vector allocation is valid*/
  info[0] = (mat->dX == NULL);
  info[1] = myrow;
  info[2] = mycol;
  (void)HPL_all_reduce((void*)(info), 3, HPL_INT, HPL_MAX, GRID->all_comm);
  if(info[0] != 0) {
    HPL_pwarn(TEST->outfp,
              __LINE__,
              "HPL_pdmatgen",
              "[%d,%d] %s",
              info[1],
              info[2],
              "Device memory allocation failed for x. Skip.");
    return HPL_FAILURE;
  }

  int Anp;
  Mnumroc(Anp, mat->n, mat->nb, mat->nb, myrow, 0, nprow);

  /*Need space for a column of panels for pdfact on CPU*/
  size_t A_hostsize = mat->ld * mat->nb * sizeof(double);

#ifdef HPL_VERBOSE_PRINT
  if((myrow == 0) && (mycol == 0)) {
    printf("Allocating %g GBs of storage on CPU...",
           ((double)A_hostsize) / (1024 * 1024 * 1024));
    fflush(stdout);
  }
#endif

  /*Need space for a column of panels for pdfact on CPU*/
  unsigned long pg_size = sysconf(_SC_PAGESIZE);
   /*Need space for a column of panels for pdfact on CPU*/
  //mat->A = (double *) malloc(A_hostsize);
  posix_memalign((void**)&(mat->A),pg_size, A_hostsize);

  // const int KB = ALGO->pfactb; //pfact blocking size
  const int KB = NB; //pfact blocking size

  #pragma omp parallel
  {
    /*First touch*/
    const int thread_rank = omp_get_thread_num();
    const int thread_size = omp_get_num_threads();
    assert(thread_size <= max_nthreads);

    for (int i=0;i<mat->ld;i+=KB) {
      if( (i/KB) % thread_size == thread_rank) {
        const int mm = std::min(KB, mat->ld-i);
        for (int k=0;k<NB;++k) {
          for (int j=0;j<mm;++j) {
            mat->A[j+static_cast<size_t>(mat->ld)*k] = 0.0;
          }
        }
      }
    }
  }

  hipHostRegister(mat->A, A_hostsize, hipHostRegisterDefault);
  // hipHostMalloc((void**)&(mat->A), A_hostsize);

  /*Check workspace allocation is valid*/
  info[0] = (mat->A == NULL);
  info[1] = myrow;
  info[2] = mycol;
  (void)HPL_all_reduce((void*)(info), 3, HPL_INT, HPL_MAX, GRID->all_comm);
  if(info[0] != 0) {
    HPL_pwarn(TEST->outfp,
              __LINE__,
              "HPL_pdmatgen",
              "[%d,%d] %s",
              info[1],
              info[2],
              "Host memory allocation failed for host A. Skip.");
    return HPL_FAILURE;
  }
#ifdef HPL_VERBOSE_PRINT
  if((myrow == 0) && (mycol == 0)) printf("done.\n");
#endif

  size_t dworkspace_size = 0;
  size_t workspace_size  = 0;

#if 0
  // determine how much workspace rocBLAS needs for DTRSM
  rocblas_start_device_memory_size_query(handle);

  // sample DTRSM call used in pdupdate
  const double one = 1.0;
  rocblas_dtrsm(handle,
                rocblas_side_right,
                rocblas_fill_lower,
                rocblas_operation_transpose,
                rocblas_diagonal_unit,
                Anq,
                mat->nb,
                &one,
                mat->dA,
                mat->ld,
                mat->dA,
                mat->ld);

  // also sample some reductions to be safe
  int id;
  rocblas_idamax(handle, mat->mp, mat->dA, 1, &id);
  rocblas_idamax(handle, mat->nq, mat->dA, 1, &id);

  size_t rocblas_workspace_size = 0;
  rocblas_stop_device_memory_size_query(handle, &rocblas_workspace_size);
  dworkspace_size = Mmax(rocblas_workspace_size, dworkspace_size);
#endif

  /*pdtrsv needs two vectors for B and W (and X on host) */
  dworkspace_size = Mmax(2 * Anp * sizeof(double), dworkspace_size);
  workspace_size  = Mmax((2 * Anp + nq) * sizeof(double), workspace_size);

  /*Scratch space for rows in pdlaswp (with extra space for padding) */
  dworkspace_size = Mmax((nq+256) * mat->nb * sizeof(double), dworkspace_size);
  workspace_size  = Mmax((nq+256) * mat->nb * sizeof(double), workspace_size);

#ifdef HPL_VERBOSE_PRINT
  if((myrow == 0) && (mycol == 0)) {
    printf("Allocating %g GBs of storage on GPU...",
           ((double)dworkspace_size) / (1024 * 1024 * 1024));
    fflush(stdout);
  }
#endif
  hipMalloc((void**)&(mat->dW), dworkspace_size);

  /*Check workspace allocation is valid*/
  info[0] = (mat->dW == NULL);
  info[1] = myrow;
  info[2] = mycol;
  (void)HPL_all_reduce((void*)(info), 3, HPL_INT, HPL_MAX, GRID->all_comm);
  if(info[0] != 0) {
    HPL_pwarn(TEST->outfp,
              __LINE__,
              "HPL_pdmatgen",
              "[%d,%d] %s",
              info[1],
              info[2],
              "Device memory allocation failed for workspace. Skip.");
    return HPL_FAILURE;
  }
#ifdef HPL_VERBOSE_PRINT
  if((myrow == 0) && (mycol == 0)) printf("done.\n");
#endif

#ifdef HPL_VERBOSE_PRINT
  if((myrow == 0) && (mycol == 0)) {
    printf("Allocating %g GBs of storage on CPU...",
           ((double)workspace_size) / (1024 * 1024 * 1024));
    fflush(stdout);
  }
#endif
  hipHostMalloc((void**)&(mat->W), workspace_size);
  // mat->W = (double*) malloc(workspace_size);
  // hipHostRegister(mat->W, workspace_size, 0);

  /*Check workspace allocation is valid*/
  info[0] = (mat->W == NULL);
  info[1] = myrow;
  info[2] = mycol;
  (void)HPL_all_reduce((void*)(info), 3, HPL_INT, HPL_MAX, GRID->all_comm);
  if(info[0] != 0) {
    HPL_pwarn(TEST->outfp,
              __LINE__,
              "HPL_pdmatgen",
              "[%d,%d] %s",
              info[1],
              info[2],
              "Host memory allocation failed for workspace. Skip.");
    return HPL_FAILURE;
  }
#ifdef HPL_VERBOSE_PRINT
  if((myrow == 0) && (mycol == 0)) printf("done.\n");
#endif

#if 0
  // tell rocBLAS to use our device workspace
  rocblas_set_workspace(handle, mat->dW, dworkspace_size);
#endif

  return HPL_SUCCESS;
}

void HPL_pdmatfree(HPL_T_pmat* mat) {

  if(mat->dA) {hipFree(mat->dA); mat->dA=nullptr;}
  if(mat->dX) {hipFree(mat->dX); mat->dX=nullptr;}
  if(mat->dW) {hipFree(mat->dW); mat->dW=nullptr;}

  if(mat->A) {free(mat->A); mat->A=nullptr;}
  if(mat->W) {hipHostFree(mat->W); mat->W=nullptr;}
    // if(mat->W) free(mat->W);

#if 0
  // tell rocblas we free'd the workspace
  rocblas_set_device_memory_size(handle, 0);
#endif

  rocblas_destroy_handle(handle);
}
