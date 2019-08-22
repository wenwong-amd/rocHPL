//print out last 384x384 block of A
      char fname[BUFSIZ];
      sprintf(fname, "HPL_%04d.bin", frame++);
      FILE *fp = fopen(fname, "w");

      hipMemcpy(A->A, A->dA, (N+1)*A->ld*sizeof(double), hipMemcpyDeviceToHost);

      double *pA  = A->A  + N-385 + (size_t)((N-385))*A->ld;
      double *dpA = A->dA + N-385 + (size_t)((N-385))*A->ld;
      for (int jj=0;jj<384;jj++) {
         for (int ii=0;ii<384;ii++) {
            fprintf(fp, "%a\t", pA[jj + ii*A->ld]);
         }
         fprintf(fp, "\n");
      }
      fclose(fp);



      char fname[BUFSIZ];
            hipMemcpy(AA, A->dA, (N+1)*A->ld*sizeof(double), hipMemcpyDeviceToHost);

            sprintf(fname, "HPL_%04d_A_%dx%d.txt", j/jb, jb, 5760);
            FILE *fp = fopen(fname, "w");

            double *pA  = AA  + j    + (size_t)((j+jb))*A->ld;
            for (int jj=0;jj<jb;jj++) {
               for (int ii=N-j-jb-5760;ii<N-j-jb;ii++) {
                  fprintf(fp, "%.17g\t", pA[jj + ii*A->ld]);
               }
               fprintf(fp, "\n");
            }
            fclose(fp);


            sprintf(fname, "HPL_%04d_B_%dx%d.txt", j/jb, 5760, jb);
            fp = fopen(fname, "w");

            double *pB  = AA  + j+jb + (size_t)((j))*A->ld;
            for (int jj=N-j-jb-5760;jj<N-j-jb;jj++) {
               for (int ii=0;ii<jb;ii++) {
                  fprintf(fp, "%.17g\t", pB[jj + ii*A->ld]);
               }
               fprintf(fp, "\n");
            }
            fclose(fp);

            sprintf(fname, "HPL_%04d_C_%dx%d.txt", j/jb, 5760, 5760);
            fp = fopen(fname, "w");

            double *pC  = AA  + j+jb + (size_t)((j+jb))*A->ld;
            for (int jj=N-j-jb-5760;jj<N-j-jb;jj++) {
               for (int ii=N-j-jb-5760;ii<N-j-jb;ii++) {
                  fprintf(fp, "%.17g\t", pC[jj + ii*A->ld]);
               }
               fprintf(fp, "\n");
            }
            fclose(fp);




if (j/jb == 50) {
            double * AA = (double*) malloc((N+1)*A->ld*sizeof(double));
            char fname[BUFSIZ];
            hipMemcpy(AA, A->dA, (N+1)*A->ld*sizeof(double), hipMemcpyDeviceToHost);

            sprintf(fname, "HPL_%04d_A_%dx%d.txt", j/jb, jb, N-j-jb);
            FILE *fp = fopen(fname, "w");

            double *pA  = AA  + j    + (size_t)((j+jb))*A->ld;
            for (int jj=0;jj<jb;jj++) {
               for (int ii=0;ii<N-j-jb;ii++) {
                  fprintf(fp, "%.17g\t", pA[jj + ii*A->ld]);
               }
               fprintf(fp, "\n");
            }
            fclose(fp);


            sprintf(fname, "HPL_%04d_B_%dx%d.txt", j/jb, N-j-jb, jb);
            fp = fopen(fname, "w");

            double *pB  = AA  + j+jb + (size_t)((j))*A->ld;
            for (int jj=0;jj<N-j-jb;jj++) {
               for (int ii=0;ii<jb;ii++) {
                  fprintf(fp, "%.17g\t", pB[jj + ii*A->ld]);
               }
               fprintf(fp, "\n");
            }
            fclose(fp);

            sprintf(fname, "HPL_%04d_C_%dx%d.txt", j/jb, N-j-jb, N-j-jb);
            fp = fopen(fname, "w");

            double *pC  = AA  + j+jb + (size_t)((j+jb))*A->ld;
            for (int jj=0;jj<N-j-jb;jj++) {
               for (int ii=0;ii<N-j-jb;ii++) {
                  fprintf(fp, "%.17g\t", pC[jj + ii*A->ld]);
               }
               fprintf(fp, "\n");
            }
            fclose(fp);
         }
