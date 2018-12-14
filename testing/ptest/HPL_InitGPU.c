#include "hpl.h"


rocblas_handle handle;

/*
  The first time DGEMM or DTRSM are called, the library needs to map a GPU to the MPI process.
  This variable checks if this step has already been performed
*/
hipStream_t computeStream, dataStream;

int stringCmp( const void *a, const void *b)
{
  char *c_a = (char*) a;
  char *c_b = (char*) b;
  return strcmp(c_a,c_b);
}

static char     host_name[MPI_MAX_PROCESSOR_NAME];

/*
  This function finds out how many MPI processes are running on the same node
  and assigns a local rank that can be used to map a process to a device.
  This function needs to be called by all the MPI processes.
*/
void  HPL_InitGPU(){

  char (*host_names)[MPI_MAX_PROCESSOR_NAME];

  int i, n, namelen, color, rank, nprocs;
  size_t bytes;
  int dev;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Get_processor_name(host_name,&namelen);

  bytes = nprocs * sizeof(char[MPI_MAX_PROCESSOR_NAME]);
  host_names = (char (*)[MPI_MAX_PROCESSOR_NAME]) malloc(bytes);

  strcpy(host_names[rank], host_name);

  for (n=0; n<nprocs; n++){
    MPI_Bcast(&(host_names[n]),MPI_MAX_PROCESSOR_NAME, MPI_CHAR, n, MPI_COMM_WORLD);
  }

  int localRank = 0;
  for (n=0; n<rank; n++){
    if (!strcmp(host_name, host_names[n])) localRank++;
  }

  int localSize = 0;
  for (n=0; n<nprocs; n++){
    if (!strcmp(host_name, host_names[n])) localSize++;
  }

  /* Find out how many DP capable GPUs are in the system and their device number */
  hipInit(0);

  int deviceCount;
  hipGetDeviceCount(&deviceCount);

#ifdef VERBOSE_PRINT
  printf ("Assigning device %d on node %s to rank %d \n", localRank%deviceCount,  host_name, rank);
#endif

  /* Assign device to MPI process, initialize BLAS and probe device properties */
  dev = localRank%deviceCount;
  hipSetDevice(dev);

  rocblas_create_handle(&handle);

  hipStreamCreate(&computeStream);
  hipStreamCreate(&dataStream);
}


void  Free_gpu(){

  rocblas_destroy_handle(handle);

  hipStreamDestroy(computeStream);
  hipStreamDestroy(dataStream);
}
