//
// Created by andre on 2/03/23.
//

#ifndef SVD_THESIS_LIB_GLOBAL_CUH_
#define SVD_THESIS_LIB_GLOBAL_CUH_

#define NT 16
//#define DEBUG
//#define SEQUENTIAL
//#define REPORT
//#define OMP
//#define LAPACK
//#define CUDA
#define CUDA_KERNEL
//#define CUDA_KERNEL_GPU
//#define IMKL
#define TESTS

#define CHECK_CUBLAS(call)                                                     \
{                                                                              \
    cublasStatus_t err;                                                        \
    if ((err = (call)) != CUBLAS_STATUS_SUCCESS)                               \
    {                                                                          \
        fprintf(stderr, "Got CUBLAS error %d at %s:%d\n", err, __FILE__,       \
                __LINE__);                                                     \
        exit(1);                                                               \
    }                                                                          \
}

#define CHECK_CUDA(call)                                                     \
{                                                                              \
    cudaError_t err;                                                        \
    if ((err = (call)) != cudaSuccess)                               \
    {                                                                          \
        fprintf(stderr, "Got CUDA error %d at %s:%d\n", err, __FILE__,       \
                __LINE__);                                                   \
        fprintf(stderr, "Got CUDA error %s\n", cudaGetErrorString(err));                                                   \
        exit(1);                                                               \
    }                                                                          \
}

#define iteratorR(i,j,ld)(((i)*(ld))+(j))
#define iteratorC(i,j,ld)(((j)*(ld))+(i))

// For double precision accuracy in the eigenvalues and eigenvectors, a tolerance of order 10−16 will suffice. Erricos
#define tolerance 1e-16

#endif //SVD_THESIS_LIB_GLOBAL_CUH_
