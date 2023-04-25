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
#define CUDA
//#define TESTS

// For double precision accuracy in the eigenvalues and eigenvectors, a tolerance of order 10−16 will suffice. Erricos
#define tolerance 1e-16

#endif //SVD_THESIS_LIB_GLOBAL_CUH_
