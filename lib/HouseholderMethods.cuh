//
// Created by andre on 2/03/23.
//

#ifndef SVD_THESIS_LIB_HOUSEHOLDERMETHODS_CUH_
#define SVD_THESIS_LIB_HOUSEHOLDERMETHODS_CUH_

struct Matrix;
struct CUDAMatrix;

#include <fstream>
#include <iostream>
#include <iomanip>
#include "global.cuh"
#include "Matrix.cuh"
#include "Utils.cuh"

namespace Thesis {

void QRDecompositionParallelWithB(const Matrix &A, Matrix &B, Matrix &R, std::ofstream &file);

void QRDecompositionParallel(const Matrix &A, Matrix &R, std::ofstream &file);

__global__ void CUDAInitializeHouseholderVector(unsigned long index_column,
                                                CUDAMatrix d_R,
                                                unsigned long R_height,
                                                unsigned long R_width,
                                                double *d_v,
                                                unsigned long d_v_size,
                                                double *d_w,
                                                unsigned long d_w_size);

__global__ void CUDAQRDecompositionOnIndexColumn(double beta1,
                                                 unsigned long index_column,
                                                 CUDAMatrix d_R,
                                                 unsigned long R_height,
                                                 unsigned long R_width,
                                                 double *d_v,
                                                 unsigned long d_v_size,
                                                 double *d_w,
                                                 unsigned long d_w_size);

__global__ void test_CUDAQRDecompositionOnIndexColumn(unsigned long index_column,
                                                      CUDAMatrix d_R,
                                                      unsigned long R_height,
                                                      unsigned long R_width);

void QRDecompositionCUDA(const Matrix &A, Matrix &R, std::ofstream &file);

void dgeqrf_gpu(size_t m,
                size_t n,
                Matrix &dA,
                size_t ldda,
                Matrix &tau,
                Matrix &dt,
                int info);

/* -------------------------------- Householder Bidiagonalization Methods -------------------------------- */

void sequential_pdgebrd(size_t m,
                       size_t n,
                       MATRIX_LAYOUT matrix_layout_A,
                       Matrix &A,
                       size_t lda,
                       Matrix &s,
                       Matrix &U,
                       size_t ldu,
                       Matrix &V,
                       size_t ldv);

void blas_pdgebrd(size_t m,
                 size_t n,
                 MATRIX_LAYOUT matrix_layout_A,
                 const Matrix &A,
                 size_t lda,
                 Matrix &s,
                 Matrix &U,
                 size_t ldu,
                 Matrix &V,
                 size_t ldv);

void omp_pdgebrd(size_t m,
                size_t n,
                MATRIX_LAYOUT matrix_layout_A,
                const Matrix &A,
                size_t lda,
                Matrix &s,
                Matrix &U,
                size_t ldu,
                Matrix &V,
                size_t ldv);

void cuda_pdgebrd(size_t m,
                 size_t n,
                 MATRIX_LAYOUT matrix_layout_A,
                 const Matrix &A,
                 size_t lda,
                 Matrix &s,
                 Matrix &U,
                 size_t ldu,
                 Matrix &V,
                 size_t ldv);

}

#endif //SVD_THESIS_LIB_HOUSEHOLDERMETHODS_CUH_
