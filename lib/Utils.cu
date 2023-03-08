//
// Created by andre on 7/03/23.
//

#include "Utils.cuh"

namespace Thesis {
/***************************************************************************
      Purpose
      -------
      non_sym_Schur returns the tuple [c,s] to make a Jacobi rotation in
      A^T * A of a matrix A of m x n and index (p,q), with 1 ≤ p < q ≤ n-1.

      Arguments
      ---------
      @param[in]
      m       SIZE_T
              The number of rows of the input matrix A.  M >= 0.

      @param[in]
      n       SIZE_T
              The number of columns of the input matrix A.  N >= 0.

      @param[in]
      A       DOUBLE PRECISION array, dimension (LDA,N)
              On entry, the M-by-N matrix A.

      @param[in]
      lda     SIZE_T
              The leading dimension of the array A. Depends in the ordering

      @param[in]
      p     SIZE_T
              The index p of row to eliminate in A.

      @param[in]
      q     SIZE_T
              The index q of column to eliminate in A.

      @param[out]
      info    std::tuple<double, double>
              The tuple [c,s] that eliminates the element a_{p,q} in
              A^T * A

      Note: We expect A[p,q] != 0.0
      Reference: Handbook of Parallel Computing and Statistics, Erricos, 2006 by Taylor & Francis Group p. 128
 *********************************************************************************/
std::tuple<double, double> non_sym_Schur(MATRIX_LAYOUT matrix_layout,
                                         size_t m,
                                         size_t n,
                                         const Matrix &A,
                                         size_t lda,
                                         size_t p,
                                         size_t q) {
  if (!(1 <= p < q <= n)) {
    throw std::runtime_error(
        "index (p,q) is not 1 <= p < q <= n with index (" + std::to_string(p) + "," + std::to_string(q) + ").");
  }

  double alpha = 0.0;
  double beta = 0.0;
  double gamma = 0.0;
  double c = 0.0, s = 0.0;

  if (matrix_layout == ROW_MAJOR) {
    for (size_t index_row = 0; index_row < m; ++index_row) {
      alpha += A.elements[IteratorR(index_row, p, lda)] * A.elements[IteratorR(index_row, q, lda)];
      beta += A.elements[IteratorR(index_row, q, lda)] * A.elements[IteratorR(index_row, q, lda)];
    }
  } else {
    for (size_t index_row = 0; index_row < m; ++index_row) {
      alpha += A.elements[IteratorC(index_row, p, lda)] * A.elements[IteratorC(index_row, q, lda)];
      beta += A.elements[IteratorC(index_row, q, lda)] * A.elements[IteratorC(index_row, q, lda)];
    }
  }

  alpha = 2.0 * alpha;
  gamma = sqrt(alpha * alpha + beta * beta);

  if (beta > 0.0) {
    c = sqrt((beta + gamma) / (2.0 * gamma));
    s = alpha / (2.0 * gamma * c);
  } else {
    s = sqrt((gamma - beta) / (2.0 * gamma));
    c = alpha / (2.0 * gamma * s);
  }
  return std::make_tuple(c, s);
}
} // Thesis