//
// Created by andre on 4/18/23.
//

#include "Tests.cuh"

void Thesis::max_iterations_error(){
  // SEED!!!
  const unsigned seed = 1000000;
  omp_set_num_threads(16);

  for(auto matrix_size = 4000; matrix_size <= 5000; matrix_size += 1000){
    std::cout << "size: " << matrix_size << "\n";
    for(auto maximum_iterations = 1; maximum_iterations <= 6; maximum_iterations += 1){
      {
        std::cout << "maximum iterations: " << maximum_iterations << "\n";
        auto height = matrix_size;
        auto width = matrix_size;
        Matrix A(height, width), U(height, height), V(width, width), s(1, std::min(A.height, A.width)), A_copy(height, width);

        std::fill_n(U.elements, matrix_size * matrix_size, 0.0);
        std::fill_n(V.elements, matrix_size * matrix_size, 0.0);
        std::fill_n(A.elements, matrix_size * matrix_size, 0.0);
        std::fill_n(A_copy.elements, matrix_size * matrix_size, 0.0);
        std::fill_n(s.elements, matrix_size, 0.0);

        // Select iterator
        auto iterator = Thesis::IteratorC;

        // Create a random bidiagonal matrix
        std::default_random_engine e(seed);
        std::uniform_real_distribution<double> uniform_dist(1.0, 2.0);
        for (size_t indexRow = 0; indexRow < matrix_size; ++indexRow) {
          double value = uniform_dist(e);
          A.elements[iterator(indexRow, indexRow, matrix_size)] = value;
          A_copy.elements[iterator(indexRow, indexRow, matrix_size)] = value;
        }

        for (size_t indexRow = 0; indexRow < (matrix_size - 1); ++indexRow) {
          double value = uniform_dist(e);
          A.elements[iterator(indexRow, indexRow + 1, matrix_size)] = value;
          A_copy.elements[iterator(indexRow, indexRow + 1, matrix_size)] = value;
        }

        // Initializing V = 1
        for (size_t i = 0; i < matrix_size; ++i) {
          V.elements[iterator(i, i, matrix_size)] = 1.0;
        }

        // Stopping condition in Hogben, L. (Ed.). (2013). Handbook of Linear Algebra (2nd ed.). Chapman and Hall/CRC. https://doi.org/10.1201/b16113
        size_t m = matrix_size;
        size_t n = matrix_size;
        size_t lda = matrix_size;
        size_t ldv = matrix_size;
        size_t ldu = matrix_size;
        size_t m_ordering = (n + 1) / 2;
        size_t istop = 0;
        size_t stop_condition = matrix_size * (matrix_size - 1) / 2;
        uint16_t reps = 0;
        uint16_t maxIterations = 5;

        do {
          istop = 0;
          // Ordering in  A. Sameh. On Jacobi and Jacobi-like algorithms for a parallel computer. Math. Comput., 25:579–590,
          // 1971
          for (size_t k = 1; k < m_ordering; ++k) {
            size_t p = 0;
            size_t p_trans = 0;
            size_t q_trans = 0;
#pragma omp parallel for private(p, p_trans, q_trans)
            for (size_t q = m_ordering - k + 1; q <= n - k; ++q) {
              if (m_ordering - k + 1 <= q && q <= (2 * m_ordering) - (2 * k)) {
                p = ((2 * m_ordering) - (2 * k) + 1) - q;
              } else if ((2 * m_ordering) - (2 * k) < q && q <= (2 * m_ordering) - k - 1) {
                p = ((4 * m_ordering) - (2 * k)) - q;
              } else if ((2 * m_ordering) - k - 1 < q) {
                p = n;
              }

              // Translate to (0,0)
              p_trans = p - 1;
              q_trans = q - 1;

              double alpha = 0.0, beta = 0.0, gamma = 0.0;
              // \alpha = a_p^T\cdot a_q, \beta = a_p^T\cdot a_p, \gamma = a_q^T\cdot a_q
              double tmp_p, tmp_q;
              for (size_t i = 0; i < m; ++i) {
                tmp_p = A.elements[iterator(i, p_trans, lda)];
                tmp_q = A.elements[iterator(i, q_trans, lda)];
                alpha += tmp_p * tmp_q;
                beta += tmp_p * tmp_p;
                gamma += tmp_q * tmp_q;
              }

              // abs(a_p^T\cdot a_q) / sqrt((a_p^T\cdot a_p)(a_q^T\cdot a_q))
              double convergence_value = abs(alpha) / sqrt(beta * gamma);

              if (convergence_value > tolerance) {

                // Schur
                double c_schur = 1.0, s_schur = 0.0, aqq = 0.0, app = 0.0, apq = alpha;

                // Calculate a_{pp}, a_{qq}, a_{pq}
                for(size_t i = 0; i < m; ++i){
                  double value_p = A.elements[iterator(i, p, lda)];
                  double value_q = A.elements[iterator(i, q, lda)];
                  app += value_p * value_p;
                  aqq += value_q * value_q;
                }

                if(abs(apq) > tolerance){
                  double tau = (aqq - app) / (2.0 * apq);
                  double t = 0.0;

                  if(tau >= 0){
                    t = 1.0 / (tau + sqrt(1 + (tau * tau)));
                  } else {
                    t = 1.0 / (tau - sqrt(1 + (tau * tau)));
                  }

                  c_schur = 1.0 / sqrt(1 + (t * t));
                  s_schur = t * c_schur;
                }

                double tmp_A_p, tmp_A_q;
                for (size_t i = 0; i < m; ++i) {
                  tmp_A_p = A.elements[iterator(i, p_trans, lda)];
                  tmp_A_q = A.elements[iterator(i, q_trans, lda)];
                  tmp_p = c_schur * tmp_A_p - s_schur * tmp_A_q;
                  tmp_q = s_schur * tmp_A_p + c_schur * tmp_A_q;
                  A.elements[iterator(i, p_trans, lda)] = tmp_p;
                  A.elements[iterator(i, q_trans, lda)] = tmp_q;
                }

                for (size_t i = 0; i < n; ++i) {
                  tmp_p = c_schur * V.elements[iterator(i, p_trans, ldv)] - s_schur * V.elements[iterator(i, q_trans, ldv)];
                  tmp_q = s_schur * V.elements[iterator(i, p_trans, ldv)] + c_schur * V.elements[iterator(i, q_trans, ldv)];
                  V.elements[iterator(i, p_trans, ldv)] = tmp_p;
                  V.elements[iterator(i, q_trans, ldv)] = tmp_q;
                }

              } else {
                ++istop;
              }
            }
          }

          for (size_t k = m_ordering; k < 2 * m_ordering; ++k) {
            size_t p = 0;
            size_t p_trans = 0;
            size_t q_trans = 0;
#pragma omp parallel for private(p, p_trans, q_trans)
            for (size_t q = (4 * m_ordering) - n - k; q < (3 * m_ordering) - k; ++q) {
              if (q < (2 * m_ordering) - k + 1) {
                p = n;
              } else if ((2 * m_ordering) - k + 1 <= q && q <= (4 * m_ordering) - (2 * k) - 1) {
                p = ((4 * m_ordering) - (2 * k)) - q;
              } else if ((4 * m_ordering) - (2 * k) - 1 < q) {
                p = ((6 * m_ordering) - (2 * k) - 1) - q;
              }

              // Translate to (0,0)
              p_trans = p - 1;
              q_trans = q - 1;

              double alpha = 0.0, beta = 0.0, gamma = 0.0;
              // \alpha = a_p^T\cdot a_q, \beta = a_p^T\cdot a_p, \gamma = a_q^T\cdot a_q
              double tmp_p, tmp_q;
              for (size_t i = 0; i < m; ++i) {
                tmp_p = A.elements[iterator(i, p_trans, lda)];
                tmp_q = A.elements[iterator(i, q_trans, lda)];
                alpha += tmp_p * tmp_q;
                beta += tmp_p * tmp_p;
                gamma += tmp_q * tmp_q;
              }

              // (a_p^T\cdot a_q)^2 / (a_p^T\cdot a_p)(a_q^T\cdot a_q)
              double convergence_value = abs(alpha) / sqrt(beta * gamma);

              if (convergence_value > tolerance) {
                // (a_p^T\cdot a_q)^2 / (a_p^T\cdot a_p)(a_q^T\cdot a_q) > tolerance
                // Schur
                double c_schur = 1.0, s_schur = 0.0, aqq = 0.0, app = 0.0, apq = alpha;

                // Calculate a_{pp}, a_{qq}, a_{pq}
                for(size_t i = 0; i < m; ++i){
                  double value_p = A.elements[iterator(i, p, lda)];
                  double value_q = A.elements[iterator(i, q, lda)];
                  app += value_p * value_p;
                  aqq += value_q * value_q;
                }

                if(abs(apq) > tolerance){
                  double tau = (aqq - app) / (2.0 * apq);
                  double t = 0.0;

                  if(tau >= 0){
                    t = 1.0 / (tau + sqrt(1 + (tau * tau)));
                  } else {
                    t = 1.0 / (tau - sqrt(1 + (tau * tau)));
                  }

                  c_schur = 1.0 / sqrt(1 + (t * t));
                  s_schur = t * c_schur;
                }

                double A_tmp_p, A_tmp_q;
                for (size_t i = 0; i < m; ++i) {
                  A_tmp_p = A.elements[iterator(i, p_trans, lda)];
                  A_tmp_q = A.elements[iterator(i, q_trans, lda)];
                  tmp_p = c_schur * A_tmp_p - s_schur * A_tmp_q;
                  tmp_q = s_schur * A_tmp_p + c_schur * A_tmp_q;
                  A.elements[iterator(i, p_trans, lda)] = tmp_p;
                  A.elements[iterator(i, q_trans, lda)] = tmp_q;
                }

                for (size_t i = 0; i < n; ++i) {
                  tmp_p = c_schur * V.elements[iterator(i, p_trans, ldv)] - s_schur * V.elements[iterator(i, q_trans, ldv)];
                  tmp_q = s_schur * V.elements[iterator(i, p_trans, ldv)] + c_schur * V.elements[iterator(i, q_trans, ldv)];
                  V.elements[iterator(i, p_trans, ldv)] = tmp_p;
                  V.elements[iterator(i, q_trans, ldv)] = tmp_q;
                }
              } else {
                ++istop;
              }
            }
          }
        } while (++reps < maxIterations);

        // Compute \Sigma
#pragma omp parallel for
        for (size_t k = 0; k < matrix_size; ++k) {
          for (size_t i = 0; i < m; ++i) {
            s.elements[k] += A.elements[iterator(i, k, lda)] * A.elements[iterator(i, k, lda)];
          }
          s.elements[k] = sqrt(s.elements[k]);
        }

        //Compute U
#pragma omp parallel for
        for (size_t i = 0; i < m; ++i) {
          for (size_t j = 0; j < m; ++j) {
            U.elements[iterator(j, i, ldu)] = A.elements[iterator(j, i, ldu)] / s.elements[i];
          }
        }

        double maxError = 0.0;
        for (size_t indexRow = 0; indexRow < matrix_size; ++indexRow) {
          for (size_t indexCol = 0; indexCol < matrix_size; ++indexCol) {
            double value = 0.0;
            for (size_t k_dot = 0; k_dot < matrix_size; ++k_dot) {
              value += U.elements[iterator(indexRow, k_dot, matrix_size)] * s.elements[k_dot]
                  * V.elements[iterator(indexCol, k_dot, matrix_size)];
            }
            double diff = std::abs(A_copy.elements[iterator(indexRow, indexCol, matrix_size)] - value);
            maxError = std::max<double>(maxError, diff);
          }
        }

        std::cout << "max error between A and USV: " << maxError << "\n";
      }
    }
  }
}

//void Thesis::lapack_svd_times(){
//
//}