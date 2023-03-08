//
// Created by andre on 7/03/23.
//

#ifndef SVD_THESIS_LIB_UTILS_CUH_
#define SVD_THESIS_LIB_UTILS_CUH_

#include "Matrix.cuh"
#include "global.cuh"
#include <tuple>
#include <stdexcept>

namespace Thesis {

std::tuple<double, double> non_sym_Schur(MATRIX_LAYOUT matrix_layout,size_t m, size_t n, const Matrix &A, size_t lda, size_t p, size_t q);

} // Thesis

#endif //SVD_THESIS_LIB_UTILS_CUH_
