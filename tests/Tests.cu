//
// Created by andre on 4/18/23.
//

#include "Tests.cuh"

namespace Thesis{
/*
void max_iterations_error(){
  // SEED!!!
  const unsigned seed = 1000000;
  omp_set_num_threads(16);

  for(auto matrix_size = 100; matrix_size <= 100; matrix_size += 100){
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
void compare_cuda_operations(){
  {
    std::cout
        << "-------------------------------- Test dot product --------------------------------\n";

    // Select iterator
    auto iterator = Thesis::IteratorC;
    // Create a handle for CUBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);

    Matrix s(1, 500);

    // Create a random symmetric matrix
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> uniform_dist(0.0, 1.0);
    for (size_t indexRow = 0; indexRow < 500; ++indexRow) {
      double value = uniform_dist(gen);
      s.elements[indexRow] = value;
    }

    CUDAMatrix d_s(s);

    double d_dot;
    cublasDdot (handle, 500,
                reinterpret_cast<const double *>(d_s.elements), 1,
                reinterpret_cast<const double *>(d_s.elements), 1,
                &d_dot);

    double dot = 0.0;
    for (size_t indexRow = 0; indexRow < 500; ++indexRow) {
      double element = s.elements[indexRow];
      dot += element * element;
    }

    std::cout << "dot: " << dot << "\n";
    std::cout << "d_dot: " << d_dot << "\n";

    cublasDestroy_v2(handle);
  }

  {
    std::cout
        << "-------------------------------- Test matrix column dot product --------------------------------\n";

    // Select iterator
    auto iterator = Thesis::IteratorC;
    // Create a handle for CUBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);

    Matrix A(10, 10);

    // Create a random symmetric matrix
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> uniform_dist(0.0, 1.0);
    for (size_t indexRow = 0; indexRow < 10; ++indexRow) {
      for (size_t indexCol = indexRow; indexCol < 10; ++indexCol) {
        double value = uniform_dist(gen);
        A.elements[iterator(indexRow, indexCol, 10)] = value;
        A.elements[iterator(indexCol, indexRow, 10)] = value;
      }
    }

    CUDAMatrix d_A(A);

    double d_dot;
    cublasDdot (handle, 10,
                reinterpret_cast<const double *>(d_A.elements + 3 * 10), 1,
                reinterpret_cast<const double *>(d_A.elements + 4 * 10), 1,
                &d_dot);

    double dot = 0.0;
    double tmp_p, tmp_q;
    for (size_t i = 0; i < 10; ++i) {
      tmp_p = A.elements[iterator(i, 3, 10)];
      tmp_q = A.elements[iterator(i, 4, 10)];
      dot += tmp_p * tmp_q;
    }

    std::cout << "dot: " << dot << "\n";
    std::cout << "d_dot: " << d_dot << "\n";

    d_A.free();
    cublasDestroy_v2(handle);
  }

  {
    // Select iterator
    auto iterator = Thesis::IteratorC;
    // Create a handle for CUBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);

    std::cout << std::fixed << std::setprecision(3)
        << "-------------------------------- Test Givens rotation --------------------------------\n";
    Matrix A(500, 500), A_copy(500,500);

    // Create a random symmetric matrix
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> uniform_dist(0.0, 1.0);
    for (size_t indexRow = 0; indexRow < 500; ++indexRow) {
      for (size_t indexCol = indexRow; indexCol < 500; ++indexCol) {
        double value = uniform_dist(gen);
        A.elements[iterator(indexRow, indexCol, 500)] = value;
        A.elements[iterator(indexCol, indexRow, 500)] = value;
      }
    }

    CUDAMatrix d_A(A);

    d_A.copy_to_host(A_copy);

    for(auto i = 0; i < 10; ++i){
      std::random_device rd_int;
      std::mt19937 gen_int(rd_int());
      std::uniform_int_distribution<> uniform_int_dist(0, 99);

      size_t p = uniform_int_dist(gen_int);
      size_t q = uniform_int_dist(gen_int);

      double alpha = 0.0, beta = 0.0, gamma = 0.0;
      // \alpha = a_p^T\cdot a_q, \beta = a_p^T\cdot a_p, \gamma = a_q^T\cdot a_q
      double tmp_p, tmp_q;
      for (size_t j = 0; j < 500; ++j) {
        tmp_p = A.elements[iterator(j, p, 500)];
        tmp_q = A.elements[iterator(j, q, 500)];
        alpha += tmp_p * tmp_q;
        beta += tmp_p * tmp_p;
        gamma += tmp_q * tmp_q;
      }

      double d_alpha = 0.0, d_beta = 0.0, d_gamma = 0.0;
      // \alpha = a_p^T\cdot a_q, \beta = a_p^T\cdot a_p, \gamma = a_q^T\cdot a_q
      cublasDdot (handle, 500,
                  reinterpret_cast<const double *>(d_A.elements + (500 * p)), 1,
                  reinterpret_cast<const double *>(d_A.elements + (500 * q)), 1,
                  &d_alpha);
      cublasDdot (handle, 500,
                  reinterpret_cast<const double *>(d_A.elements + (500 * p)), 1,
                  reinterpret_cast<const double *>(d_A.elements + (500 * p)), 1,
                  &d_beta);
      cublasDdot (handle, 500,
                  reinterpret_cast<const double *>(d_A.elements + 500 * q), 1,
                  reinterpret_cast<const double *>(d_A.elements + 500 * q), 1,
                  &d_gamma);

      if(abs(alpha - d_alpha) > 1e-6){
        std::cout << "cublasDdot different from manual dot product!!\n";
        std::cout << "alpha: " << alpha << "\n";
        std::cout << "d_alpha: " << d_alpha << "\n";
        break;
      }

      if(abs(beta - d_beta) > 1e-6){
        std::cout << "cublasDdot different from manual dot product!!\n";
        std::cout << "beta: " << beta << "\n";
        std::cout << "d_beta: " << d_beta << "\n";
        break;
      }

      if(abs(gamma - d_gamma) > 1e-6){
        std::cout << "cublasDdot different from manual dot product!!\n";
        std::cout << "gamma: " << gamma << "\n";
        std::cout << "d_gamma: " << d_gamma << "\n";
        break;
      }

      double tau = (gamma - beta) / (2.0 * alpha);
      double t = 0.0;

      if(tau >= 0){
        t = 1.0 / (tau + sqrt(1 + (tau * tau)));
      } else {
        t = 1.0 / (tau - sqrt(1 + (tau * tau)));
      }

      const double c_schur = 1.0 / sqrt(1 + (t * t));
      const double s_schur = t * c_schur;

      cublasDrot(handle, 500,
                 d_A.elements + (500 * q), 1,
                 d_A.elements + (500 * p), 1,
                 &c_schur, &s_schur);

      double A_tmp_p, A_tmp_q;
      for (size_t j = 0; j < 500; ++j) {
        A_tmp_p = A.elements[iterator(j, p, 500)];
        A_tmp_q = A.elements[iterator(j, q, 500)];
        tmp_p = c_schur * A_tmp_p - s_schur * A_tmp_q;
        tmp_q = s_schur * A_tmp_p + c_schur * A_tmp_q;
        A.elements[iterator(j, p, 500)] = tmp_p;
        A.elements[iterator(j, q, 500)] = tmp_q;
      }

      d_A.copy_to_host(A_copy);

      // Compare matrices
      double maxError = 0.0;
      for (size_t indexRow = 0; indexRow < A.height; ++indexRow) {
        for (size_t indexCol = 0; indexCol < A.width; ++indexCol) {
          double diff = std::abs(A_copy.elements[iterator(indexRow, indexCol, 500)] - A.elements[iterator(indexRow, indexCol, 500)]);
          maxError = std::max<double>(maxError, diff);
        }
      }

      std::cout << "max error between A and a_copy: " << maxError << "\n";
    }

    // Destroy the handle
    cublasDestroy(handle);
    d_A.free();
  }

  {
    // Select iterator
    auto iterator = Thesis::IteratorC;
    // Create a handle for CUBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);

    std::cout << std::fixed << std::setprecision(3)
              << "-------------------------------- Test Givens rotation --------------------------------\n";
    Matrix A(500, 500), A_copy(500,500);

    // Create a random symmetric matrix
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> uniform_dist(0.0, 1.0);
    for (size_t indexRow = 0; indexRow < 500; ++indexRow) {
      for (size_t indexCol = indexRow; indexCol < 500; ++indexCol) {
        double value = uniform_dist(gen);
        A.elements[iterator(indexRow, indexCol, 500)] = value;
        A.elements[iterator(indexCol, indexRow, 500)] = value;
      }
    }

    CUDAMatrix d_A(A);

    d_A.copy_to_host(A_copy);

    // Jacobi on device
    // Stopping condition in Hogben, L. (Ed.). (2013). Handbook of Linear Algebra (2nd ed.). Chapman and Hall/CRC. https://doi.org/10.1201/b16113
    size_t m = 500;
    size_t n = 500;
    size_t m_ordering = (n + 1) / 2;

    // Ordering in  A. Sameh. On Jacobi and Jacobi-like algorithms for a parallel computer. Math. Comput., 25:579–590,
    // 1971
    for (size_t k = 1; k < m_ordering; ++k) {
      size_t p = 0;
      size_t p_trans = 0;
      size_t q_trans = 0;
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
        cublasDdot (handle, m,
                    reinterpret_cast<const double *>(d_A.elements + m * p_trans), 1,
                    reinterpret_cast<const double *>(d_A.elements + m * q_trans), 1,
                    &alpha);
        cublasDdot (handle, m,
                    reinterpret_cast<const double *>(d_A.elements + m * p_trans), 1,
                    reinterpret_cast<const double *>(d_A.elements + m * p_trans), 1,
                    &beta);
        cublasDdot (handle, m,
                    reinterpret_cast<const double *>(d_A.elements + m * q_trans), 1,
                    reinterpret_cast<const double *>(d_A.elements + m * q_trans), 1,
                    &gamma);

        // abs(a_p^T\cdot a_q) / sqrt((a_p^T\cdot a_p)(a_q^T\cdot a_q))
        double convergence_value = abs(alpha) / sqrt(beta * gamma);

        if (convergence_value > tolerance) {
          // Schur
          if(abs(alpha) > tolerance){
            double tau = (gamma - beta) / (2.0 * alpha);
            double t = 0.0;

            if(tau >= 0){
              t = 1.0 / (tau + sqrt(1 + (tau * tau)));
            } else {
              t = 1.0 / (tau - sqrt(1 + (tau * tau)));
            }

            const double c_schur = 1.0 / sqrt(1 + (t * t));
            const double s_schur = t * c_schur;

            cublasDrot(handle, m,
                       d_A.elements + m * q_trans, 1,
                       d_A.elements + m * p_trans, 1,
                       &c_schur, &s_schur);
          }
        }
      }
    }

    for (size_t k = m_ordering; k < 2 * m_ordering; ++k) {
      size_t p = 0;
      size_t p_trans = 0;
      size_t q_trans = 0;
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
        cublasDdot (handle, m,
                    reinterpret_cast<const double *>(d_A.elements + m * p_trans), 1,
                    reinterpret_cast<const double *>(d_A.elements + m * q_trans), 1,
                    &alpha);
        cublasDdot (handle, m,
                    reinterpret_cast<const double *>(d_A.elements + m * p_trans), 1,
                    reinterpret_cast<const double *>(d_A.elements + m * p_trans), 1,
                    &beta);
        cublasDdot (handle, m,
                    reinterpret_cast<const double *>(d_A.elements + m * q_trans), 1,
                    reinterpret_cast<const double *>(d_A.elements + m * q_trans), 1,
                    &gamma);

        // abs(a_p^T\cdot a_q) / sqrt((a_p^T\cdot a_p)(a_q^T\cdot a_q))
        double convergence_value = abs(alpha) / sqrt(beta * gamma);

        if (convergence_value > tolerance) {
          // Schur
          if(abs(alpha) > tolerance){
            double tau = (gamma - beta) / (2.0 * alpha);
            double t = 0.0;

            if(tau >= 0){
              t = 1.0 / (tau + sqrt(1 + (tau * tau)));
            } else {
              t = 1.0 / (tau - sqrt(1 + (tau * tau)));
            }

            const double c_schur = 1.0 / sqrt(1 + (t * t));
            const double s_schur = t * c_schur;

            cublasDrot(handle, m,
                       d_A.elements + m * q_trans, 1,
                       d_A.elements + m * p_trans, 1,
                       &c_schur, &s_schur);
          }
        }
      }
    }

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
          tmp_p = A.elements[iterator(i, p_trans, 500)];
          tmp_q = A.elements[iterator(i, q_trans, 500)];
          alpha += tmp_p * tmp_q;
          beta += tmp_p * tmp_p;
          gamma += tmp_q * tmp_q;
        }

        // abs(a_p^T\cdot a_q) / sqrt((a_p^T\cdot a_p)(a_q^T\cdot a_q))
        double convergence_value = abs(alpha) / sqrt(beta * gamma);

        // Schur
//        auto [c_schur, s_schur] = non_sym_Schur_non_ordered(iterator, m, n, A, 500, p_trans, q_trans, alpha);

        if (convergence_value > tolerance) {
          // Schur
          if (abs(alpha) > tolerance) {
            double tau = (gamma - beta) / (2.0 * alpha);
            double t = 0.0;

            if(tau >= 0){
              t = 1.0 / (tau + sqrt(1 + (tau * tau)));
            } else {
              t = 1.0 / (tau - sqrt(1 + (tau * tau)));
            }

            const double c_schur = 1.0 / sqrt(1 + (t * t));
            const double s_schur = t * c_schur;

            double tmp_A_p, tmp_A_q;
            for (size_t i = 0; i < m; ++i) {
              tmp_A_p = A.elements[iterator(i, p_trans, 500)];
              tmp_A_q = A.elements[iterator(i, q_trans, 500)];
              tmp_p = c_schur * tmp_A_p - s_schur * tmp_A_q;
              tmp_q = s_schur * tmp_A_p + c_schur * tmp_A_q;
              A.elements[iterator(i, p_trans, 500)] = tmp_p;
              A.elements[iterator(i, q_trans, 500)] = tmp_q;
            }
          }
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
          tmp_p = A.elements[iterator(i, p_trans, 500)];
          tmp_q = A.elements[iterator(i, q_trans, 500)];
          alpha += tmp_p * tmp_q;
          beta += tmp_p * tmp_p;
          gamma += tmp_q * tmp_q;
        }

        // (a_p^T\cdot a_q)^2 / (a_p^T\cdot a_p)(a_q^T\cdot a_q)
        double convergence_value = abs(alpha) / sqrt(beta * gamma);

//        if (convergence_value > tolerance) {
        // (a_p^T\cdot a_q)^2 / (a_p^T\cdot a_p)(a_q^T\cdot a_q) > tolerance
        // Schur
//        auto [c_schur, s_schur] = non_sym_Schur_non_ordered(iterator, m, n, A, 500, p_trans, q_trans, alpha);

        if (convergence_value > tolerance) {
          // Schur
          if (abs(alpha) > tolerance) {
            double tau = (gamma - beta) / (2.0 * alpha);
            double t = 0.0;

            if(tau >= 0){
              t = 1.0 / (tau + sqrt(1 + (tau * tau)));
            } else {
              t = 1.0 / (tau - sqrt(1 + (tau * tau)));
            }

            const double c_schur = 1.0 / sqrt(1 + (t * t));
            const double s_schur = t * c_schur;

            double A_tmp_p, A_tmp_q;
            for (size_t i = 0; i < m; ++i) {
              A_tmp_p = A.elements[iterator(i, p_trans, 500)];
              A_tmp_q = A.elements[iterator(i, q_trans, 500)];
              tmp_p = c_schur * A_tmp_p - s_schur * A_tmp_q;
              tmp_q = s_schur * A_tmp_p + c_schur * A_tmp_q;
              A.elements[iterator(i, p_trans, 500)] = tmp_p;
              A.elements[iterator(i, q_trans, 500)] = tmp_q;
            }
          }
        }
      }
    }

    d_A.copy_to_host(A_copy);

    // Compare matrices
    double maxError = 0.0;
    for (size_t indexRow = 0; indexRow < A.height; ++indexRow) {
      for (size_t indexCol = 0; indexCol < A.width; ++indexCol) {
        double diff = std::abs(A_copy.elements[iterator(indexRow, indexCol, 500)] - A.elements[iterator(indexRow, indexCol, 500)]);
        maxError = std::max<double>(maxError, diff);
      }
    }

    std::cout << "max error between A and a_copy: " << maxError << "\n";

    // Destroy the handle
    cublasDestroy(handle);
    d_A.free();
  }
  {
    std::cout << std::fixed << std::setprecision(3)
              << "-------------------------------- Test SVD  --------------------------------\n";

    const size_t height = 500;
    const size_t width = 500;

    std::cout << "Dimensions, height: " << height << ", width: " << width << "\n";

    Matrix A(height, width), U(height, height), V(width, width), s(1, std::min(A.height, A.width)), A_copy(height, width),
          A_cuda_result(height, width), U_cuda_result(height, height), V_cuda_result(width, width), s_cuda_result(1, std::min(A.height, A.width));

    const unsigned long A_height = A.height, A_width = A.width;

    std::fill_n(U.elements, U.height * U.width, 0.0);
    std::fill_n(V.elements, V.height * V.width, 0.0);
    std::fill_n(A.elements, A.height * A.width, 0.0);
    std::fill_n(A_copy.elements, A_copy.height * A_copy.width, 0.0);

    // Create a random matrix
//    std::default_random_engine e(seed);
//    std::uniform_real_distribution<double> uniform_dist(1.0, 2.0);
//    for (size_t indexRow = 0; indexRow < A_height; ++indexRow) {
//      for (size_t indexCol = 0; indexCol < A_width; ++indexCol) {
//        double value = uniform_dist(e);
//        A.elements[Thesis::IteratorC(indexRow, indexCol, A_height)] = value;
//      }
//    }

    // Select iterator
    auto iterator = Thesis::IteratorC;

    // Create a random bidiagonal matrix
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> uniform_dist(0.0, 1.0);
    for (size_t indexRow = 0; indexRow < std::min<size_t>(A_height, A_width); ++indexRow) {
      double value = uniform_dist(gen);
      A.elements[iterator(indexRow, indexRow, A_height)] = value;
      A_copy.elements[iterator(indexRow, indexRow, A_height)] = value;
    }

    for (size_t indexRow = 0; indexRow < (std::min<size_t>(A_height, A_width) - 1); ++indexRow) {
      double value = uniform_dist(gen);
      A.elements[iterator(indexRow, indexRow + 1, A_height)] = value;
      A_copy.elements[iterator(indexRow, indexRow + 1, A_height)] = value;
    }

    for (size_t i = 0; i < A.width; ++i) {
      V.elements[iterator(i, i, A_width)] = 1.0;
    }

    CUDAMatrix d_A( A), d_s(s), d_U(U), d_V(V);
    std::cout << "Initialized!!\n";

    // Calculate SVD decomposition
    double ti = omp_get_wtime();
    Thesis::cuda_dgesvd(Thesis::AllVec,
                        Thesis::AllVec,
                        A.height,
                        A.width,
                        d_A,
                        A_height,
                        d_s,
                        d_U,
                        A_height,
                        d_V,
                        A_width);
    double tf = omp_get_wtime();
    double time = tf - ti;

    // Calculate SVD decomposition
    ti = omp_get_wtime();
    Thesis::omp_dgesvd(Thesis::AllVec,
                       Thesis::AllVec,
                       A.height,
                       A.width,
                       Thesis::COL_MAJOR,
                       A,
                       A_height,
                       s,
                       U,
                       A_height,
                       V,
                       A_width);
    tf = omp_get_wtime();
    time = tf - ti;

    std::cout << "SVD OMP time with U,V calculation: " << time << "\n";
    std::cout << "SVD CUDA time with U,V calculation: " << time << "\n";

    d_A.copy_to_host(A_cuda_result), d_s.copy_to_host(s_cuda_result), d_U.copy_to_host(U_cuda_result), d_V.copy_to_host(V_cuda_result);

    double maxError = 0.0;
    for (size_t indexRow = 0; indexRow < A_height; ++indexRow) {
      for (size_t indexCol = 0; indexCol < A_width; ++indexCol) {
        double diff = std::abs(A_cuda_result.elements[iterator(indexRow, indexCol, A_height)] - A.elements[iterator(indexRow, indexCol, A_height)]);
        maxError = std::max<double>(maxError, diff);
      }
    }

    std::cout << "max error between A OMP and A CUDA: " << maxError << "\n";

    maxError = 0.0;
    for (size_t indexRow = 0; indexRow < height; ++indexRow) {
      for (size_t indexCol = 0; indexCol < height; ++indexCol) {
        double diff = std::abs(U.elements[iterator(indexRow, indexCol, A_height)] - U_cuda_result.elements[iterator(indexRow, indexCol, A_height)]);
        maxError = std::max<double>(maxError, diff);
      }
    }

    std::cout << "max error between U OMP and U CUDA: " << maxError << "\n";

    maxError = 0.0;
    for (size_t indexRow = 0; indexRow < width; ++indexRow) {
      for (size_t indexCol = 0; indexCol < width; ++indexCol) {
        double diff = std::abs(V.elements[iterator(indexRow, indexCol, A_height)] - V_cuda_result.elements[iterator(indexRow, indexCol, A_height)]);
        maxError = std::max<double>(maxError, diff);
      }
    }

    std::cout << "max error between V OMP and V CUDA: " << maxError << "\n";

    maxError = 0.0;
    for (size_t indexCol = 0; indexCol < std::min(A.height, A.width); ++indexCol) {
      double diff = std::abs(s_cuda_result.elements[indexCol] - s.elements[indexCol]);
      maxError = std::max<double>(maxError, diff);
    }

    std::cout << "max error between s OMP and s CUDA: " << maxError << "\n";

    maxError = 0.0;
    for (size_t indexRow = 0; indexRow < A_height; ++indexRow) {
      for (size_t indexCol = 0; indexCol < A_width; ++indexCol) {
        double value = 0.0;
        for (size_t k_dot = 0; k_dot < A_width; ++k_dot) {
          value += U.elements[iterator(indexRow, k_dot, A_height)] * s.elements[k_dot]
              * V.elements[iterator(indexCol, k_dot, A_height)];
        }
        double diff = std::abs(A_copy.elements[iterator(indexRow, indexCol, A_height)] - value);
        maxError = std::max<double>(maxError, diff);
      }
    }

    std::cout << "max error between A and USV: " << maxError << "\n";

    d_A.free(), d_s.free(), d_U.free(), d_V.free();
  }
}
*/

void compare_times_dot_product() {
  for(size_t i = 5000; i <= 50000; i += 5000){
    {
      std::cout << "vector size: " << i << '\n';

      Matrix a(1,i), b(1,i);
      std::random_device random_device;
      std::uniform_real_distribution<double> d(0.0,1.0);
      for(size_t index_v = 0; index_v < i; ++index_v){
        a.elements[index_v] = d(random_device);
        b.elements[index_v] = d(random_device);
      }

      double dot_product = 0.0;

      double ti = omp_get_wtime();
      #pragma omp parallel for reduction(+:dot_product)
      for(size_t index_v = 0; index_v < i; ++index_v){
        double valueA = a.elements[index_v];
        double valueB = b.elements[index_v];

        dot_product += valueA * valueB;
      }
      double tf = omp_get_wtime();
      double time = tf - ti;

      std::cout << "omp_time: " << time << '\n';
      std::cout << "result: " << dot_product << '\n';


      cublasHandle_t handle;
      cublasCreate(&handle);
      double cuda_dot_product = 0.0;

      double ti_cuda = omp_get_wtime();
      CUDAMatrix d_A(a), d_B(b);
      cublasDdot(handle, i,
                 reinterpret_cast<const double *>(d_A.elements), 1,
                 reinterpret_cast<const double *>(d_B.elements), 1,
                 &cuda_dot_product);
      d_A.copy_to_host(a), d_B.copy_to_host(b);
      double tf_cuda = omp_get_wtime();
      double cuda_time = tf_cuda - ti_cuda;

      std::cout << "cuda_time: " << cuda_time << '\n';
      std::cout << "result: " << cuda_dot_product << '\n';

      cublasDestroy(handle);
      d_A.free(), d_B.free();
    }
  }
}

void compare_times_jacobi_matrix_product() {
  for(size_t i = 5000; i <= 50000; i += 5000){
    {
      std::cout << "vector size: " << i << '\n';

      Matrix a(1,i), b(1,i);
      std::random_device random_device;
      std::uniform_real_distribution<double> d(0.0,1.0);
      for(size_t index_v = 0; index_v < i; ++index_v){
        a.elements[index_v] = d(random_device);
        b.elements[index_v] = d(random_device);
      }

      double alpha = 0.0, beta = 0.0, gamma = 0.0;
      // \alpha = a_p^T\cdot a_q, \beta = a_p^T\cdot a_p, \gamma = a_q^T\cdot a_q
      double tmp_p, tmp_q;
      for (size_t index_v = 0; index_v < i; ++index_v) {
        tmp_p = a.elements[index_v];
        tmp_q = b.elements[index_v];
        alpha += tmp_p * tmp_q;
        beta += tmp_p * tmp_p;
        gamma += tmp_q * tmp_q;
      }

      double tau = (gamma - beta) / (2.0 * alpha);
      double t = 0.0;

      if (tau >= 0) {
        t = 1.0 / (tau + sqrt(1 + (tau * tau)));
      } else {
        t = 1.0 / (tau - sqrt(1 + (tau * tau)));
      }

      const double c_schur = 1.0 / sqrt(1 + (t * t));
      const double s_schur = t * c_schur;

      double ti = omp_get_wtime();
      double tmp_A_p = 0.0, tmp_A_q = 0.0;
      for (size_t index_v = 0; index_v < i; ++index_v) {
        tmp_A_p = a.elements[index_v];
        tmp_A_q = b.elements[index_v];
        tmp_p = c_schur * tmp_A_p - s_schur * tmp_A_q;
        tmp_q = s_schur * tmp_A_p + c_schur * tmp_A_q;
        a.elements[index_v] = tmp_p;
        b.elements[index_v] = tmp_q;
      }
      double tf = omp_get_wtime();
      double time = tf - ti;

      std::cout << "omp_time: " << time << '\n';

      for(size_t index_v = 0; index_v < i; ++index_v){
        a.elements[index_v] = d(random_device);
        b.elements[index_v] = d(random_device);
      }

      alpha = 0.0, beta = 0.0, gamma = 0.0;
      // \alpha = a_p^T\cdot a_q, \beta = a_p^T\cdot a_p, \gamma = a_q^T\cdot a_q
      tmp_p = 0.0, tmp_q = 0.0;
      for (size_t index_v = 0; index_v < i; ++index_v) {
        tmp_p = a.elements[index_v];
        tmp_q = b.elements[index_v];
        alpha += tmp_p * tmp_q;
        beta += tmp_p * tmp_p;
        gamma += tmp_q * tmp_q;
      }

      tau = (gamma - beta) / (2.0 * alpha);
      t = 0.0;

      if (tau >= 0) {
        t = 1.0 / (tau + sqrt(1 + (tau * tau)));
      } else {
        t = 1.0 / (tau - sqrt(1 + (tau * tau)));
      }

      const double cuda_c_schur = 1.0 / sqrt(1 + (t * t));
      const double cuda_s_schur = t * cuda_c_schur;


      cublasHandle_t handle;
      cublasCreate(&handle);

      double ti_cuda = omp_get_wtime();
      CUDAMatrix d_A(a), d_B(b);
      cublasDrot(handle, i,
                 d_A.elements, 1,
                 d_B.elements, 1,
                 &cuda_c_schur, &cuda_s_schur);
      d_A.copy_to_host(a), d_B.copy_to_host(b);
      double tf_cuda = omp_get_wtime();
      double cuda_time = tf_cuda - ti_cuda;

      std::cout << "cuda_time: " << cuda_time << '\n';

      cublasDestroy(handle);
      d_A.free(), d_B.free();
    }
  }
}

void correctness_jacobi_kernel() {
  std::cout << "Correctness of Jacobi Kernel\n";
  for(auto size_of_vector = 1000; size_of_vector <= 10000; size_of_vector += 1000){
    size_t number_of_tests = 100;
    std::cout << "size of array: " << size_of_vector << ", number of test: " << number_of_tests << '\n';
    for(auto index_test = 0; index_test < number_of_tests; ++index_test){
      Matrix p_vector(1, size_of_vector), q_vector(1, size_of_vector);

      /* --------------------------------- Initialize array ---------------------------------------*/
      std::random_device random_device;
      std::mt19937 mt_19937(random_device());
      std::uniform_real_distribution<double> distribution(0.0, 1.0);

      std::generate(p_vector.elements,p_vector.elements + size_of_vector, [&distribution, &mt_19937](){ return distribution(mt_19937); });
      std::generate(q_vector.elements,q_vector.elements + size_of_vector, [&distribution, &mt_19937](){ return distribution(mt_19937); });

      double alpha = 0.0, beta = 0.0, gamma = 0.0;
      // \alpha = a_p^T\cdot a_q, \beta = a_p^T\cdot a_p, \gamma = a_q^T\cdot a_q
      double tmp_p, tmp_q;
      for (size_t i = 0; i < size_of_vector; ++i) {
        tmp_p = p_vector.elements[i];
        tmp_q = q_vector.elements[i];
        alpha += tmp_p * tmp_q;
        beta += tmp_p * tmp_p;
        gamma += tmp_q * tmp_q;
      }

      // Schur
      double c_schur = 1.0, s_schur = 0.0, aqq = gamma, app = beta, apq = alpha;

      if (abs(apq) > tolerance) {
//        std::cout << "Rotation initalized\n";
        double tau = (aqq - app) / (2.0 * apq);
        double t = 0.0;

        if (tau >= 0) {
          t = 1.0 / (tau + sqrt(1 + (tau * tau)));
        } else {
          t = 1.0 / (tau - sqrt(1 + (tau * tau)));
        }

        c_schur = 1.0 / sqrt(1 + (t * t));
        s_schur = t * c_schur;

        CUDAMatrix d_p_vector(p_vector), d_q_vector(q_vector);

        /* --------------------------------- Compute jacobi rotation in host ---------------------------------------*/
        double tmp_A_p, tmp_A_q;
        for (size_t i = 0; i < size_of_vector; ++i) {
          tmp_A_p = p_vector.elements[i];
          tmp_A_q = q_vector.elements[i];
          tmp_p = c_schur * tmp_A_p - s_schur * tmp_A_q;
          tmp_q = s_schur * tmp_A_p + c_schur * tmp_A_q;
          p_vector.elements[i] = tmp_p;
          q_vector.elements[i] = tmp_q;
        }

        /* --------------------------------- Compute jacobi rotation in device ---------------------------------------*/
        int threadsPerBlock = 16;
        dim3 blocksPerGrid  (ceil( float(size_of_vector) / threadsPerBlock ));
        jacobi_rotation<<<blocksPerGrid, threadsPerBlock>>>(size_of_vector, d_p_vector.elements, d_q_vector.elements, c_schur, s_schur);
        cudaDeviceSynchronize();

        Matrix p_copy(1, size_of_vector), q_copy(1, size_of_vector);

        d_p_vector.copy_to_host(p_copy);
        d_q_vector.copy_to_host(q_copy);

        /* --------------------------------- Compute frobenius norm ---------------------------------------*/

        // Calculate frobenius norm
        double p_frobenius_norm = 0.0;
        for (size_t i = 0; i < size_of_vector; ++i) {
          double value = p_copy.elements[i] - p_vector.elements[i];
          p_frobenius_norm += value*value;
        }

        if(p_frobenius_norm > 1e-16)
          std::cout << "||p - p||_F: " << sqrt(p_frobenius_norm) << "\n";

        d_p_vector.free();
        d_q_vector.free();
      }
    }
  }
}

void upload_download_correctness_jacobi_kernel(size_t height, size_t width) {
  Matrix A(height, width), A_copy(height, width);
  // Create R matrix
  std::random_device random_device;
  std::mt19937 mt_19937(random_device());
  std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);
  for (size_t indexRow = 0; indexRow < height; ++indexRow) {
    for (size_t indexCol = 0; indexCol < width; ++indexCol) {
      double value = uniform_dist(mt_19937);
      A.elements[iteratorC(indexRow, indexCol, height)] = value;
      A_copy.elements[iteratorC(indexRow, indexCol, height)] = 0.0;
    }
  }

  std::cout << "Correctness of Jacobi Kernel\n";
  double *d_a;
  cudaMalloc(&d_a, height * sizeof(double ));
  for(auto indexCol = 0; indexCol < width; indexCol++){
    cudaMemcpy(d_a, (A.elements + indexCol*height), height * sizeof(double),
               cudaMemcpyHostToDevice);
//    CUDAMatrix d_vector(A.elements + indexCol * height, height);
    cudaMemcpy((A_copy.elements + indexCol*height), d_a, height * sizeof(double),
               cudaMemcpyDeviceToHost);
//    d_vector.copy_to_host(A_copy.elements + indexCol * height, height);
//
//    d_vector.free();
  }
  cudaFree(d_a);

  // Calculate frobenius norm
  double frobenius_norm = 0.0;

  #pragma omp parallel for reduction(+:frobenius_norm)
  for (size_t indexRow = 0; indexRow < height; ++indexRow) {
    for (size_t indexCol = 0; indexCol < width; ++indexCol) {
      double value = A_copy.elements[iteratorC(indexRow, indexCol, height)] - A.elements[iteratorC(indexRow, indexCol, height)];
      frobenius_norm += value*value;
    }
  }

  std::cout << "||A-USVt||_F: " << sqrt(frobenius_norm) << "\n";
}

void jacobi_kernel_time_comparison() {
  std::cout << "Comparison times of Jacobi Kernel\n";
  for(auto size_of_vector = 10000; size_of_vector <= 50000; size_of_vector += 10000){
    size_t number_of_tests = 100;
    std::cout << "size of array: " << size_of_vector << ", number of test: " << number_of_tests << '\n';

    double kernel_time_avg = 0.0;
    double upload_time_avg = 0.0;
    double download_time_avg = 0.0;
    int threadsPerBlock = 16;
    dim3 blocksPerGrid  (ceil( float(size_of_vector) / threadsPerBlock ));

    for(auto index_test = 0; index_test < number_of_tests; ++index_test){
      Matrix p_vector(1, size_of_vector), q_vector(1, size_of_vector);

      /* --------------------------------- Initialize array ---------------------------------------*/
      std::random_device random_device;
      std::mt19937 mt_19937(random_device());
      std::uniform_real_distribution<double> distribution(0.0, 1.0);

      std::generate(p_vector.elements,p_vector.elements + size_of_vector, [&distribution, &mt_19937](){ return distribution(mt_19937); });
      std::generate(q_vector.elements,q_vector.elements + size_of_vector, [&distribution, &mt_19937](){ return distribution(mt_19937); });

      double alpha = 0.0, beta = 0.0, gamma = 0.0;
      // \alpha = a_p^T\cdot a_q, \beta = a_p^T\cdot a_p, \gamma = a_q^T\cdot a_q
      double tmp_p, tmp_q;
      for (size_t i = 0; i < size_of_vector; ++i) {
        tmp_p = p_vector.elements[i];
        tmp_q = q_vector.elements[i];
        alpha += tmp_p * tmp_q;
        beta += tmp_p * tmp_p;
        gamma += tmp_q * tmp_q;
      }

      // Schur
      double c_schur = 1.0, s_schur = 0.0, aqq = gamma, app = beta, apq = alpha;

      if (abs(apq) > tolerance) {
//        std::cout << "Rotation initalized\n";
        double tau = (aqq - app) / (2.0 * apq);
        double t = 0.0;

        if (tau >= 0) {
          t = 1.0 / (tau + sqrt(1 + (tau * tau)));
        } else {
          t = 1.0 / (tau - sqrt(1 + (tau * tau)));
        }

        c_schur = 1.0 / sqrt(1 + (t * t));
        s_schur = t * c_schur;

        double malloc_time_init = omp_get_wtime();
        CUDAMatrix d_p_vector(p_vector), d_q_vector(q_vector);
        double malloc_time_final = omp_get_wtime();

        upload_time_avg += (malloc_time_final - malloc_time_init);

        /* --------------------------------- Compute jacobi rotation in host ---------------------------------------*/
        double tmp_A_p, tmp_A_q;
        for (size_t i = 0; i < size_of_vector; ++i) {
          tmp_A_p = p_vector.elements[i];
          tmp_A_q = q_vector.elements[i];
          tmp_p = c_schur * tmp_A_p - s_schur * tmp_A_q;
          tmp_q = s_schur * tmp_A_p + c_schur * tmp_A_q;
          p_vector.elements[i] = tmp_p;
          q_vector.elements[i] = tmp_q;
        }

        /* --------------------------------- Compute jacobi rotation in device ---------------------------------------*/
        double jacobi_rotation_time_init = omp_get_wtime();
        jacobi_rotation<<<blocksPerGrid, threadsPerBlock>>>(size_of_vector, d_p_vector.elements, d_q_vector.elements, c_schur, s_schur);
        cudaDeviceSynchronize();
        double jacobi_rotation_time_final = omp_get_wtime();

        kernel_time_avg += (jacobi_rotation_time_final - jacobi_rotation_time_init);

        Matrix p_copy(1, size_of_vector), q_copy(1, size_of_vector);

        double device_to_host_time_init = omp_get_wtime();
        d_p_vector.copy_to_host(p_copy);
        d_q_vector.copy_to_host(q_copy);
        double device_to_host_time_final = omp_get_wtime();

        download_time_avg += (device_to_host_time_final - device_to_host_time_init);

        /* --------------------------------- Compute frobenius norm ---------------------------------------*/

        // Calculate frobenius norm
        double p_frobenius_norm = 0.0;
        for (size_t i = 0; i < size_of_vector; ++i) {
          double value = p_copy.elements[i] - p_vector.elements[i];
          p_frobenius_norm += value*value;
        }

        if(p_frobenius_norm > 1e-16)
          std::cout << "||p - p||_F: " << sqrt(p_frobenius_norm) << "\n";

        d_p_vector.free();
        d_q_vector.free();
      }
    }

    std::cout << "upload time average: " << upload_time_avg << '\n';
    std::cout << "kernel time average: " << kernel_time_avg << '\n';
    std::cout << "download time average: " << download_time_avg << '\n';

  }
}
}