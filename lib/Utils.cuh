//
// Created by andre on 7/03/23.
//

#ifndef SVD_THESIS_LIB_UTILS_CUH_
#define SVD_THESIS_LIB_UTILS_CUH_

#include "Matrix.cuh"
#include "global.cuh"
#include <tuple>
#include <stdexcept>
#include <functional>

namespace Thesis {

enum MATRIX_LAYOUT{
  ROW_MAJOR,
  COL_MAJOR
};

std::tuple<double, double> non_sym_Schur(MATRIX_LAYOUT matrix_layout,
                                         size_t m,
                                         size_t n,
                                         const Matrix &A,
                                         size_t lda,
                                         size_t p,
                                         size_t q,
                                         double alpha,
                                         double beta);

size_t IteratorC(size_t i, size_t j, size_t ld);
size_t IteratorR(size_t i, size_t j, size_t ld);

std::function<size_t(size_t, size_t, size_t)> get_iterator(MATRIX_LAYOUT matrix_layout);

} // Thesis

#endif //SVD_THESIS_LIB_UTILS_CUH_
