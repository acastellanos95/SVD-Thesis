//
// Created by andre on 2/03/23.
//

#ifndef SVD_THESIS_LIB_JACOBIMETHODS_CUH_
#define SVD_THESIS_LIB_JACOBIMETHODS_CUH_

#include <iostream>
#include <iomanip>
#include <omp.h>
#include <vector>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "Matrix.cuh"
#include "global.cuh"
#include "Utils.cuh"

namespace Thesis {

enum SVD_OPTIONS {
  AllVec,
  SomeVec,
  NoVec
};

#define V_OPTION

void sequential_dgesvd(SVD_OPTIONS jobu,
                       SVD_OPTIONS jobv,
                       size_t m,
                       size_t n,
                       MATRIX_LAYOUT matrix_layout_A,
                       Matrix &A,
                       size_t lda,
                       Matrix &s,
                       Matrix &U,
                       size_t ldu,
                       Matrix &V,
                       size_t ldv);
void blas_dgesvd(SVD_OPTIONS jobu,
                 SVD_OPTIONS jobv,
                 size_t m,
                 size_t n,
                 MATRIX_LAYOUT matrix_layout_A,
                 const Matrix &A,
                 size_t lda,
                 Matrix &s,
                 Matrix &U,
                 size_t ldu,
                 Matrix &V,
                 size_t ldv);
void omp_dgesvd(SVD_OPTIONS jobu,
                SVD_OPTIONS jobv,
                size_t m,
                size_t n,
                MATRIX_LAYOUT matrix_layout_A,
                Matrix &A,
                size_t lda,
                Matrix &s,
                Matrix &U,
                size_t ldu,
                Matrix &V,
                size_t ldv);

void cuda_dgesvd(SVD_OPTIONS jobu,
                 SVD_OPTIONS jobv,
                 size_t m,
                 size_t n,
                 CUDAMatrix &A,
                 size_t lda,
                 CUDAMatrix &s,
                 CUDAMatrix &U,
                 size_t ldu,
                 CUDAMatrix &V,
                 size_t ldv);

void cuda_streams_dgesvd(SVD_OPTIONS jobu,
                         SVD_OPTIONS jobv,
                         size_t m,
                         size_t n,
                         CUDAMatrix &A,
                         size_t lda,
                         CUDAMatrix &s,
                         CUDAMatrix &U,
                         size_t ldu,
                         CUDAMatrix &V,
                         size_t ldv);

void cuda_dgesvd_kernel(SVD_OPTIONS jobu,
                        SVD_OPTIONS jobv,
                        size_t m,
                        size_t n,
                        Matrix &A,
                        size_t lda,
                        Matrix &s,
                        Matrix &V,
                        size_t ldv);

void cuda_dgesvd_kernel(SVD_OPTIONS jobu,
                 SVD_OPTIONS jobv,
                 size_t m,
                 size_t n,
                 CUDAMatrix &A,
                 size_t lda,
                 CUDAMatrix &s,
                 CUDAMatrix &V,
                 size_t ldv);

void cuda_dgesvd_kernel_streams(SVD_OPTIONS jobu,
                        SVD_OPTIONS jobv,
                        size_t m,
                        size_t n,
                        Matrix &A,
                        size_t lda,
                        Matrix &s,
                        Matrix &V,
                        size_t ldv);

__global__ void jacobi_rotation(unsigned int n, double *x, double *y, double c, double s);
}

#endif //SVD_THESIS_LIB_JACOBIMETHODS_CUH_
