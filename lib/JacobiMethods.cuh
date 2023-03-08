//
// Created by andre on 2/03/23.
//

#ifndef SVD_THESIS_LIB_JACOBIMETHODS_CUH_
#define SVD_THESIS_LIB_JACOBIMETHODS_CUH_

#include "Matrix.cuh"
#include "global.cuh"

namespace Thesis {

enum SVD_OPTIONS {
  AllVec,
  SomeVec,
  NoVec
};

void sequential_dgesvd(SVD_OPTIONS jobu,
                       SVD_OPTIONS jobv,
                       size_t m,
                       size_t n,
                       const Matrix &A,
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
                const Matrix &A,
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
                 const Matrix &A,
                 size_t lda,
                 Matrix &s,
                 Matrix &U,
                 size_t ldu,
                 Matrix &V,
                 size_t ldv);
}

#endif //SVD_THESIS_LIB_JACOBIMETHODS_CUH_
