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
      p       SIZE_T
              The index p of row to eliminate in A.

      @param[in]
      q       SIZE_T
              The index q of column to eliminate in A.

      @param[in]
      alpha   DOUBLE
              Expecting \alpha = a_p^T\cdot a_q.

      @param[in]
      beta       DOUBLE
              Expecting \beta = a_q^T\cdot a_q.

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
                                         size_t q,
                                         double alpha,
                                         double beta) {

  alpha = 2.0 * alpha;
  beta = beta;
  double gamma = sqrt(alpha * alpha + beta * beta);
  double c = 0.0, s = 0.0;

  if (beta > 0.0) {
    c = sqrt((beta + gamma) / (2.0 * gamma));
    s = alpha / (2.0 * gamma * c);
  } else {
    s = sqrt((gamma - beta) / (2.0 * gamma));
    c = alpha / (2.0 * gamma * s);
  }
  return std::make_tuple(c, s);
}

size_t IteratorC(size_t i, size_t j, size_t ld){
  return (((j)*(ld))+(i));
}

size_t IteratorR(size_t i, size_t j, size_t ld){
  return (((i)*(ld))+(j));
}

std::function<size_t(size_t, size_t, size_t)> get_iterator(MATRIX_LAYOUT matrix_layout){
  if(matrix_layout == ROW_MAJOR)
    return IteratorR;
  else
    return IteratorC;
}

} // Thesis