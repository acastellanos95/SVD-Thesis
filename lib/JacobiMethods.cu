//
// Created by andre on 2/03/23.
//

#include "JacobiMethods.cuh"

namespace Thesis {

/***************************************************************************
    Purpose
    -------
    sequential_dgesvd computes the singular value decomposition (SVD)
    of a real M-by-N with m>>n matrix A using Jacobi one sided
    algorithm with no parallelism, optionally computing the left
    and/or right singular vectors. The SVD is written like

        A = U * SIGMA * transpose(V)

    where SIGMA is an M-by-N matrix which is zero except for its
    min(m,n) diagonal elements, U is an M-by-M orthogonal matrix, and
    V is an N-by-N orthogonal matrix.  The diagonal elements of SIGMA
    are the singular values of A; they are real and non-negative, and
    are returned in descending order.  The first min(m,n) columns of
    U and V are the left and right singular vectors of A.

    Note on one sided Jacobi:

        V = ((IxJ_0)xJ_1,...)
        U = A\sigma^{-1}

    Note that the routine returns VT = V**T, not V.

    Arguments
    ---------
    @param[in]
    jobu    SVD_OPTIONS
            Specifies options for computing all or part of the matrix U:
      -     = AllVec:        all M columns of U are returned in array U:
      -     = SomeVec:       the first min(m,n) columns of U (the left singular
                                  vectors) are returned in the array U;
      -     = NoVec:         no columns of U (no left singular vectors) are
                                  computed.

    @param[in]
    jobvt   SVD_OPTIONS
            Specifies options for computing all or part of the matrix V**T:
      -     = AllVec:        all N rows of V**T are returned in the array VT;
      -     = SomeVec:       the first min(m,n) rows of V**T (the right singular
                                  vectors) are returned in the array VT;
      -     = NoVec:         no rows of V**T (no right singular vectors) are
                                  computed.
    \n

    @param[in]
    m       INTEGER
            The number of rows of the input matrix A.  M >= 0.

    @param[in]
    n       INTEGER
            The number of columns of the input matrix A.  N >= 0.

    @param[in]
    matrix_layout_A MATRIX_LAYOUT
            The layout of the matrix A. It can only be
            ROW_MAJOR or COL_MAJOR.

    @param[in,out]
    A       DOUBLE PRECISION array, dimension (M,N)
            On entry, the M-by-N matrix A.

    @param[in]
    lda     INTEGER
            The leading dimension of the array A.

    @param[out]
    s       DOUBLE PRECISION array, dimension (min(M,N))
            The singular values of A, sorted so that S(i) >= S(i+1).

    @param[out]
    U       DOUBLE PRECISION array in major column order, dimension (LDU,UCOL)
            (LDU,M) if JOBU = AllVec or (LDU,min(M,N)) if JOBU = SomeVec.
      -     If JOBU = AllVec, U contains the M-by-M orthogonal matrix U;
      -     if JOBU = SomeVec, U contains the first min(m,n) columns of U
            (the left singular vectors, stored columnwise);
      -     if JOBU = NoVec, U is not referenced.

    @param[in]
    ldu     INTEGER
            The leading dimension of the array U.  LDU >= 1; if
            JOBU = SomeVec or AllVec, LDU >= M.

    @param[out]
    V      DOUBLE PRECISION array in major column order, dimension (LDV,N)
      -     If JOBVT = AllVec, VT contains the N-by-N orthogonal matrix V**T;
      -     if JOBVT = SomeVec, VT contains the first min(m,n) rows of V**T
            (the right singular vectors, stored rowwise);
      -     if JOBVT = NoVec, VT is not referenced.

    @param[in]
    ldv    INTEGER
            The leading dimension of the array VT.  LDVT >= 1;
      -     if JOBVT = AllVec, LDVT >= N;
      -     if JOBVT = SomeVec , LDVT >= min(M,N).

    @param[out]
    work    (workspace) DOUBLE PRECISION array, dimension (MAX(1,LWORK))
            On exit, if INFO = 0, WORK[0] returns the required LWORK.
            if INFO > 0, WORK(2:MIN(M,N)) contains the unconverged
            superdiagonal elements of an upper bidiagonal matrix B
            whose diagonal is in S (not necessarily sorted). B
            satisfies A = U * B * VT, so it has the same singular values
            as A, and singular vectors related by U and VT.

    @param[in]
    lwork   INTEGER
            The dimension of the array WORK.
            If lwork = -1, a workspace query is assumed.  The optimal
            size for the WORK array is calculated and stored in WORK[0],
            and no other work except argument checking is performed.
    \n
            Let mx = max(M,N) and mn = min(M,N).
            The threshold for mx >> mn is currently mx >= 1.6*mn.
            For job: N=None, O=Overwrite, S=Some, A=All.
            Paths below assume M >= N; for N > M swap jobu and jobvt.
    \n
            Because of varying nb for different subroutines, formulas below are
            an upper bound. Querying gives an exact number.
            The optimal block size nb can be obtained through magma_get_dgesvd_nb(M,N).
            For many cases, there is a fast algorithm, and a slow algorithm that
            uses less workspace. Here are sizes for both cases.
    \n
            Optimal lwork (fast algorithm)
            for mx >> mn:
            Path 1:   jobu=N, jobvt=any                  3*mn + 2*mn*nb
            Path 2:   jobu=O, jobvt=N        mn*mn +     3*mn + 2*mn*nb
                                   or        mn*mn + max(3*mn + 2*mn*nb, mn + mx*mn)
            Path 3:   jobu=O, jobvt=A,S      mn*mn +     3*mn + 2*mn*nb
                                   or        mn*mn + max(3*mn + 2*mn*nb, mn + mx*mn)
            Path 4:   jobu=S, jobvt=N        mn*mn +     3*mn + 2*mn*nb
            Path 5:   jobu=S, jobvt=O      2*mn*mn +     3*mn + 2*mn*nb
            Path 6:   jobu=S, jobvt=A,S      mn*mn +     3*mn + 2*mn*nb
            Path 7:   jobu=A, jobvt=N        mn*mn + max(3*mn + 2*mn*nb, mn + mx*nb)
            Path 8:   jobu=A, jobvt=O      2*mn*mn + max(3*mn + 2*mn*nb, mn + mx*nb)
            Path 9:   jobu=A, jobvt=A,S      mn*mn + max(3*mn + 2*mn*nb, mn + mx*nb)
            for mx >= mn, but not mx >> mn:
            Path 10:  jobu=any, jobvt=any  3*mn + (mx + mn)*nb
    \n
            Optimal lwork (slow algorithm)
            for mx >> mn:
            Path 1:   jobu=N, jobvt=any    n/a
            Path 2:   jobu=O, jobvt=N      3*mn + (mx + mn)*nb
            Path 3-9:                      3*mn + max(2*mn*nb, mx*nb)
            for mx >= mn, but not mx >> mn:
            Path 10:  jobu=any, jobvt=any  n/a
    \n
            MAGMA requires the optimal sizes above, while LAPACK has the same
            optimal sizes but the minimum sizes below.
    \n
            LAPACK minimum lwork (fast algorithm)
            for mx >> mn:
            Path 1:   jobu=N, jobvt=any              5*mn
            Path 2:   jobu=O, jobvt=N        mn*mn + 5*mn
            Path 3:   jobu=O, jobvt=A,S      mn*mn + 5*mn
            Path 4:   jobu=S, jobvt=N        mn*mn + 5*mn
            Path 5:   jobu=S, jobvt=O      2*mn*mn + 5*mn
            Path 6:   jobu=S, jobvt=A,S      mn*mn + 5*mn
            Path 7:   jobu=A, jobvt=N        mn*mn + max(5*mn, mn + mx)
            Path 8:   jobu=A, jobvt=O      2*mn*mn + max(5*mn, mn + mx)
            Path 9:   jobu=A, jobvt=A,S      mn*mn + max(5*mn, mn + mx)
            for mx >= mn, but not mx >> mn:
            Path 10:  jobu=any, jobvt=any  max(3*mn + mx, 5*mn)
    \n
            LAPACK minimum lwork (slow algorithm)
            for mx >> mn:
            Path 1:   jobu=N, jobvt=any    n/a
            Path 2-9:                      max(3*mn + mx, 5*mn)
            for mx >= mn, but not mx >> mn:
            Path 10:  jobu=any, jobvt=any  n/a

    @param[out]
    info    INTEGER
      -     = 0:  successful exit.
      -     < 0:  if INFO = -i, the i-th argument had an illegal value.
      -     > 0:  if DBDSQR did not converge, INFO specifies how many
                superdiagonals of an intermediate bidiagonal form B
                did not converge to zero. See the description of WORK
                above for details.
*********************************************************************************/
/*
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
                       size_t ldv) {

  auto iterator = get_iterator(matrix_layout_A);

  // Initializing V = 1
  if (jobv == AllVec) {
    for (size_t i = 0; i < n; ++i) {
      V.elements[iterator(i, i, ldv)] = 1.0;
    }
  } else if (jobv == SomeVec) {
    for (size_t i = 0; i < std::min(m, n); ++i) {
      V.elements[iterator(i, i, ldv)] = 1.0;
    }
  }

#ifdef DEBUG
  // Report Matrix A^T * A
  std::cout << std::fixed << std::setprecision(3) << "A^T * A: \n";
  for (size_t indexRow = 0; indexRow < m; ++indexRow) {
    for (size_t indexCol = 0; indexCol < n; ++indexCol) {
      double value = 0.0;
      for(size_t k_dot = 0; k_dot < m; ++k_dot){
        value += A.elements[iterator(k_dot, indexRow, lda)] * A.elements[iterator(k_dot, indexCol, lda)];
      }
      std::cout << value << " ";
    }
    std::cout << '\n';
  }
#endif
  // Stopping condition in Hogben, L. (Ed.). (2013). Handbook of Linear Algebra (2nd ed.). Chapman and Hall/CRC. https://doi.org/10.1201/b16113
  size_t istop = 0;
  size_t stop_condition = n * (n - 1) / 2;
  size_t m_ordering = (n + 1) / 2;
  uint16_t reps = 0;
  uint16_t maxIterations = 30;

  do {
    istop = 0;
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
        for (size_t i = 0; i < m; ++i) {
          double tmp_p = A.elements[iterator(i, p_trans, lda)];
          double tmp_q = A.elements[iterator(i, q_trans, lda)];
          alpha += tmp_p * tmp_q;
          beta += tmp_p * tmp_p;
          gamma += tmp_q * tmp_q;
        }

        // abs(a_p^T\cdot a_q) / sqrt((a_p^T\cdot a_p)(a_q^T\cdot a_q))
        double convergence_value = abs(alpha) / sqrt(beta * gamma);

        if (convergence_value > tolerance) {
          auto [c_schur, s_schur] = non_sym_Schur_non_ordered(iterator, m, n, A, lda, p_trans, q_trans, alpha);

          double tmp_p, tmp_q;
          for (size_t i = 0; i < m; ++i) {
            tmp_p = c_schur * A.elements[iterator(i, p_trans, lda)] - s_schur * A.elements[iterator(i, q_trans, lda)];
            tmp_q = s_schur * A.elements[iterator(i, p_trans, lda)] + c_schur * A.elements[iterator(i, q_trans, lda)];
            A.elements[iterator(i, p_trans, lda)] = tmp_p;
            A.elements[iterator(i, q_trans, lda)] = tmp_q;
          }

          if (jobv == AllVec || jobv == SomeVec) {
            for (size_t i = 0; i < n; ++i) {
              tmp_p = c_schur * V.elements[iterator(i, p_trans, ldv)] - s_schur * V.elements[iterator(i, q_trans, ldv)];
              tmp_q = s_schur * V.elements[iterator(i, p_trans, ldv)] + c_schur * V.elements[iterator(i, q_trans, ldv)];
              V.elements[iterator(i, p_trans, ldv)] = tmp_p;
              V.elements[iterator(i, q_trans, ldv)] = tmp_q;
            }
          }
        } else {
          ++istop;
        }

#ifdef DEBUG
        // Report Matrix A^T * A
        std::cout << std::fixed << std::setprecision(3) << "A^T * A: \n";
        for (size_t indexRow = 0; indexRow < m; ++indexRow) {
          for (size_t indexCol = 0; indexCol < n; ++indexCol) {
            double value = 0.0;
            for(size_t k_dot = 0; k_dot < m; ++k_dot){
              value += A.elements[iterator(k_dot, indexRow, lda)] * A.elements[iterator(k_dot, indexCol, lda)];
            }
            std::cout << value << " ";
          }
          std::cout << '\n';
        }
#endif
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
        for (size_t i = 0; i < m; ++i) {
          double tmp_p = A.elements[iterator(i, p_trans, lda)];
          double tmp_q = A.elements[iterator(i, q_trans, lda)];
          alpha += tmp_p * tmp_q;
          beta += tmp_p * tmp_p;
          gamma += tmp_q * tmp_q;
        }

        // (a_p^T\cdot a_q)^2 / (a_p^T\cdot a_p)(a_q^T\cdot a_q)
        double convergence_value = abs(alpha) / sqrt(beta * gamma);

        if (convergence_value > tolerance) {
          // (a_p^T\cdot a_q)^2 / (a_p^T\cdot a_p)(a_q^T\cdot a_q) > tolerance
          auto [c_schur, s_schur] = non_sym_Schur_non_ordered(iterator, m, n, A, lda, p, q_trans, alpha);
          double tmp_p, tmp_q;
          for (size_t i = 0; i < m; ++i) {
            tmp_p = c_schur * A.elements[iterator(i, p_trans, lda)] - s_schur * A.elements[iterator(i, q_trans, lda)];
            tmp_q = s_schur * A.elements[iterator(i, p_trans, lda)] + c_schur * A.elements[iterator(i, q_trans, lda)];
            A.elements[iterator(i, p_trans, lda)] = tmp_p;
            A.elements[iterator(i, q_trans, lda)] = tmp_q;
          }
          if (jobv == AllVec || jobv == SomeVec) {
            for (size_t i = 0; i < n; ++i) {
              tmp_p = c_schur * V.elements[iterator(i, p_trans, ldv)] - s_schur * V.elements[iterator(i, q_trans, ldv)];
              tmp_q = s_schur * V.elements[iterator(i, p_trans, ldv)] + c_schur * V.elements[iterator(i, q_trans, ldv)];
              V.elements[iterator(i, p_trans, ldv)] = tmp_p;
              V.elements[iterator(i, q_trans, ldv)] = tmp_q;
            }
          }
        } else {
          ++istop;
        }

#ifdef DEBUG
        // Report Matrix A^T * A
        std::cout << std::fixed << std::setprecision(3) << "A^T * A: \n";
        for (size_t indexRow = 0; indexRow < m; ++indexRow) {
          for (size_t indexCol = 0; indexCol < n; ++indexCol) {
            double value = 0.0;
            for(size_t k_dot = 0; k_dot < m; ++k_dot){
              value += A.elements[iterator(k_dot, indexRow, lda)] * A.elements[iterator(k_dot, indexCol, lda)];
            }
            std::cout << value << " ";
          }
          std::cout << '\n';
        }
#endif
      }
    }

#ifdef DEBUG
    // Report Matrix A^T * A
    std::cout << std::fixed << std::setprecision(3) << "A^T * A: \n";
    for (size_t indexRow = 0; indexRow < m; ++indexRow) {
      for (size_t indexCol = 0; indexCol < n; ++indexCol) {
        double value = 0.0;
        for(size_t k_dot = 0; k_dot < m; ++k_dot){
          value += A.elements[iterator(k_dot, indexRow, lda)] * A.elements[iterator(k_dot, indexCol, lda)];
        }
        std::cout << value << " ";
      }
      std::cout << '\n';
    }
#endif
  } while (++reps < maxIterations && istop < stop_condition);

  // Compute \Sigma
  for (size_t k = 0; k < std::min(m, n); ++k) {
    for (size_t i = 0; i < m; ++i) {
      s.elements[k] += A.elements[iterator(i, k, lda)] * A.elements[iterator(i, k, lda)];
    }
    s.elements[k] = sqrt(s.elements[k]);
  }

  //Compute U
  if (jobu == AllVec) {
    for (size_t i = 0; i < m; ++i) {
      for (size_t j = 0; j < m; ++j) {
        U.elements[iterator(j, i, ldu)] = A.elements[iterator(j, i, ldu)] / s.elements[i];
      }
    }
  } else if (jobu == SomeVec) {
    for (size_t k = 0; k < std::min(m, n); ++k) {
      for (size_t i = 0; i < m; ++i) {
        U.elements[iterator(i, k, ldu)] = A.elements[iterator(i, k, ldu)] / s.elements[k];
      }
    }
  }
}
*/

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
                size_t ldv) {

  auto iterator = get_iterator(matrix_layout_A);

  // Initializing V = 1
  if (jobv == AllVec) {
    for (size_t i = 0; i < n; ++i) {
      V.elements[iteratorC(i, i, ldv)] = 1.0;
    }
  } else if (jobv == SomeVec) {
    for (size_t i = 0; i < std::min(m, n); ++i) {
      V.elements[iteratorC(i, i, ldv)] = 1.0;
    }
  }

  size_t m_ordering = (n + 1) / 2;

#ifdef DEBUG
  // Report Matrix A^T * A
  std::cout << std::fixed << std::setprecision(3) << "A^T * A: \n";
  for (size_t indexRow = 0; indexRow < m; ++indexRow) {
    for (size_t indexCol = 0; indexCol < n; ++indexCol) {
      double value = 0.0;
      for(size_t k_dot = 0; k_dot < m; ++k_dot){
        value += A.elements[iterator(k_dot, indexRow, lda)] * A.elements[iterator(k_dot, indexCol, lda)];
      }
      std::cout << value << " ";
    }
    std::cout << '\n';
  }
#endif
  // Stopping condition in Hogben, L. (Ed.). (2013). Handbook of Linear Algebra (2nd ed.). Chapman and Hall/CRC. https://doi.org/10.1201/b16113
  size_t istop = 0;
  size_t stop_condition = n * (n - 1) / 2;
  uint16_t reps = 0;
  uint16_t maxIterations = 1;

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
          tmp_p = A.elements[iteratorC(i, p_trans, lda)];
          tmp_q = A.elements[iteratorC(i, q_trans, lda)];
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
          for (size_t i = 0; i < m; ++i) {
            double value_p = A.elements[iteratorC(i, p_trans, lda)];
            double value_q = A.elements[iteratorC(i, q_trans, lda)];
            app += value_p * value_p;
            aqq += value_q * value_q;
          }

          if (abs(apq) > tolerance) {
            double tau = (aqq - app) / (2.0 * apq);
            double t = 0.0;

            if (tau >= 0) {
              t = 1.0 / (tau + sqrt(1 + (tau * tau)));
            } else {
              t = 1.0 / (tau - sqrt(1 + (tau * tau)));
            }

            c_schur = 1.0 / sqrt(1 + (t * t));
            s_schur = t * c_schur;

            double tmp_A_p, tmp_A_q;
            for (size_t i = 0; i < m; ++i) {
              tmp_A_p = A.elements[iteratorC(i, p_trans, lda)];
              tmp_A_q = A.elements[iteratorC(i, q_trans, lda)];
              tmp_p = c_schur * tmp_A_p - s_schur * tmp_A_q;
              tmp_q = s_schur * tmp_A_p + c_schur * tmp_A_q;
              A.elements[iteratorC(i, p_trans, lda)] = tmp_p;
              A.elements[iteratorC(i, q_trans, lda)] = tmp_q;
            }

            if (jobv == AllVec || jobv == SomeVec) {
              for (size_t i = 0; i < n; ++i) {
                tmp_p =
                    c_schur * V.elements[iteratorC(i, p_trans, ldv)] - s_schur * V.elements[iteratorC(i, q_trans, ldv)];
                tmp_q =
                    s_schur * V.elements[iteratorC(i, p_trans, ldv)] + c_schur * V.elements[iteratorC(i, q_trans, ldv)];
                V.elements[iteratorC(i, p_trans, ldv)] = tmp_p;
                V.elements[iteratorC(i, q_trans, ldv)] = tmp_q;
              }
            }
          }
        }

#ifdef DEBUG
        // Report Matrix A^T * A
        std::cout << std::fixed << std::setprecision(3) << "A^T * A: \n";
        for (size_t indexRow = 0; indexRow < m; ++indexRow) {
          for (size_t indexCol = 0; indexCol < n; ++indexCol) {
            double value = 0.0;
            for(size_t k_dot = 0; k_dot < m; ++k_dot){
              value += A.elements[iterator(k_dot, indexRow, lda)] * A.elements[iterator(k_dot, indexCol, lda)];
            }
            std::cout << value << " ";
          }
          std::cout << '\n';
        }
#endif
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
          tmp_p = A.elements[iteratorC(i, p_trans, lda)];
          tmp_q = A.elements[iteratorC(i, q_trans, lda)];
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
          for (size_t i = 0; i < m; ++i) {
            double value_p = A.elements[iteratorC(i, p_trans, lda)];
            double value_q = A.elements[iteratorC(i, q_trans, lda)];
            app += value_p * value_p;
            aqq += value_q * value_q;
          }

          if (abs(apq) > tolerance) {
            double tau = (aqq - app) / (2.0 * apq);
            double t = 0.0;

            if (tau >= 0) {
              t = 1.0 / (tau + sqrt(1 + (tau * tau)));
            } else {
              t = 1.0 / (tau - sqrt(1 + (tau * tau)));
            }

            c_schur = 1.0 / sqrt(1 + (t * t));
            s_schur = t * c_schur;

            double tmp_A_p, tmp_A_q;
            for (size_t i = 0; i < m; ++i) {
              tmp_A_p = A.elements[iteratorC(i, p_trans, lda)];
              tmp_A_q = A.elements[iteratorC(i, q_trans, lda)];
              tmp_p = c_schur * tmp_A_p - s_schur * tmp_A_q;
              tmp_q = s_schur * tmp_A_p + c_schur * tmp_A_q;
              A.elements[iteratorC(i, p_trans, lda)] = tmp_p;
              A.elements[iteratorC(i, q_trans, lda)] = tmp_q;
            }

            if (jobv == AllVec || jobv == SomeVec) {
              for (size_t i = 0; i < n; ++i) {
                tmp_p =
                    c_schur * V.elements[iteratorC(i, p_trans, ldv)] - s_schur * V.elements[iteratorC(i, q_trans, ldv)];
                tmp_q =
                    s_schur * V.elements[iteratorC(i, p_trans, ldv)] + c_schur * V.elements[iteratorC(i, q_trans, ldv)];
                V.elements[iteratorC(i, p_trans, ldv)] = tmp_p;
                V.elements[iteratorC(i, q_trans, ldv)] = tmp_q;
              }
            }
          }
        }

#ifdef DEBUG
        // Report Matrix A^T * A
        std::cout << std::fixed << std::setprecision(3) << "A^T * A: \n";
        for (size_t indexRow = 0; indexRow < m; ++indexRow) {
          for (size_t indexCol = 0; indexCol < n; ++indexCol) {
            double value = 0.0;
            for(size_t k_dot = 0; k_dot < m; ++k_dot){
              value += A.elements[iterator(k_dot, indexRow, lda)] * A.elements[iterator(k_dot, indexCol, lda)];
            }
            std::cout << value << " ";
          }
          std::cout << '\n';
        }
#endif
      }
    }

#ifdef DEBUG
    // Report Matrix A^T * A
    std::cout << std::fixed << std::setprecision(3) << "A^T * A: \n";
    for (size_t indexRow = 0; indexRow < m; ++indexRow) {
      for (size_t indexCol = 0; indexCol < n; ++indexCol) {
        double value = 0.0;
        for(size_t k_dot = 0; k_dot < m; ++k_dot){
          value += A.elements[iterator(k_dot, indexRow, lda)] * A.elements[iterator(k_dot, indexCol, lda)];
        }
        std::cout << value << " ";
      }
      std::cout << '\n';
    }
#endif
  } while (++reps < maxIterations);

  std::cout << "How many repetitions?: " << reps << "\n";

  // Compute \Sigma
#pragma omp parallel for
  for (size_t k = 0; k < std::min(m, n); ++k) {
    for (size_t i = 0; i < m; ++i) {
      s.elements[k] += A.elements[iteratorC(i, k, lda)] * A.elements[iteratorC(i, k, lda)];
    }
    s.elements[k] = sqrt(s.elements[k]);
  }

  //Compute U
  if (jobu == AllVec) {
#pragma omp parallel for
    for (size_t i = 0; i < m; ++i) {
      for (size_t j = 0; j < m; ++j) {
        U.elements[iteratorC(j, i, ldu)] = A.elements[iteratorC(j, i, ldu)] / s.elements[i];
      }
    }
  } else if (jobu == SomeVec) {
#pragma omp parallel for
    for (size_t k = 0; k < std::min(m, n); ++k) {
      for (size_t i = 0; i < m; ++i) {
        U.elements[iteratorC(i, k, ldu)] = A.elements[iteratorC(i, k, ldu)] / s.elements[k];
      }
    }
  }

//  delete []ordering_array;
}

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
                 size_t ldv) {

  // Create a handle for CUBLAS
  cublasHandle_t handle;
  cublasCreate(&handle);

  // Stopping condition in Hogben, L. (Ed.). (2013). Handbook of Linear Algebra (2nd ed.). Chapman and Hall/CRC. https://doi.org/10.1201/b16113
  size_t m_ordering = (n + 1) / 2;
//  std::cout << "m_ordering: " << m_ordering << "\n";

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
      cublasDdot(handle, m,
                 reinterpret_cast<const double *>(A.elements + m * p_trans), 1,
                 reinterpret_cast<const double *>(A.elements + m * q_trans), 1,
                 &alpha);
      cublasDdot(handle, m,
                 reinterpret_cast<const double *>(A.elements + m * p_trans), 1,
                 reinterpret_cast<const double *>(A.elements + m * p_trans), 1,
                 &beta);
      cublasDdot(handle, m,
                 reinterpret_cast<const double *>(A.elements + m * q_trans), 1,
                 reinterpret_cast<const double *>(A.elements + m * q_trans), 1,
                 &gamma);

      // abs(a_p^T\cdot a_q) / sqrt((a_p^T\cdot a_p)(a_q^T\cdot a_q))
      double convergence_value = abs(alpha) / sqrt(beta * gamma);

      if (convergence_value > tolerance) {
        // Schur
        if (abs(alpha) > tolerance) {
          double tau = (gamma - beta) / (2.0 * alpha);
          double t = 0.0;

          if (tau >= 0) {
            t = 1.0 / (tau + sqrt(1 + (tau * tau)));
          } else {
            t = 1.0 / (tau - sqrt(1 + (tau * tau)));
          }

          const double c_schur = 1.0 / sqrt(1 + (t * t));
          const double s_schur = t * c_schur;

          cublasDrot(handle, m,
                     A.elements + m * q_trans, 1,
                     A.elements + m * p_trans, 1,
                     &c_schur, &s_schur);

          cublasDrot(handle, m,
                     V.elements + m * q_trans, 1,
                     V.elements + m * p_trans, 1,
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
      cublasDdot(handle, m,
                 reinterpret_cast<const double *>(A.elements + m * p_trans), 1,
                 reinterpret_cast<const double *>(A.elements + m * q_trans), 1,
                 &alpha);
      cublasDdot(handle, m,
                 reinterpret_cast<const double *>(A.elements + m * p_trans), 1,
                 reinterpret_cast<const double *>(A.elements + m * p_trans), 1,
                 &beta);
      cublasDdot(handle, m,
                 reinterpret_cast<const double *>(A.elements + m * q_trans), 1,
                 reinterpret_cast<const double *>(A.elements + m * q_trans), 1,
                 &gamma);

      // abs(a_p^T\cdot a_q) / sqrt((a_p^T\cdot a_p)(a_q^T\cdot a_q))
      double convergence_value = abs(alpha) / sqrt(beta * gamma);

      if (convergence_value > tolerance) {
        // Schur
        if (abs(alpha) > tolerance) {
          double tau = (gamma - beta) / (2.0 * alpha);
          double t = 0.0;

          if (tau >= 0) {
            t = 1.0 / (tau + sqrt(1 + (tau * tau)));
          } else {
            t = 1.0 / (tau - sqrt(1 + (tau * tau)));
          }

          const double c_schur = 1.0 / sqrt(1 + (t * t));
          const double s_schur = t * c_schur;

          cublasDrot(handle, m,
                     A.elements + m * q_trans, 1,
                     A.elements + m * p_trans, 1,
                     &c_schur, &s_schur);

          cublasDrot(handle, m,
                     V.elements + m * q_trans, 1,
                     V.elements + m * p_trans, 1,
                     &c_schur, &s_schur);
        }
      }
    }
  }

//  std::cout << "Finalized jacobi sweep!!\n";

  // Compute \Sigma
//  cudaDeviceSynchronize();
  Matrix s_copy(1, std::min(m, n));
  for (size_t k = 0; k < std::min(m, n); ++k) {
    double result;
    cublasDnrm2(handle, m, reinterpret_cast<const double *>(A.elements + m * k), 1, &result);
    s_copy.elements[k] = result;
  }
  s.copy_from_host(s_copy);
//  std::cout << "Finalized normalization!!\n";

  //Compute U
//  cudaDeviceSynchronize();
  U.copy_from_device(A);
  for (size_t i = 0; i < m; ++i) {
    double element = s_copy.elements[i];
    double scale = 1.0 / element;
    cublasDscal(handle, m, reinterpret_cast<const double *>(&scale), U.elements + m * i, 1);
  }
//  std::cout << "Finalized U!!\n";

  // Destroy the handle
  cublasDestroy(handle);
}

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
                         size_t ldv) {
  // number of streams
  size_t number_of_streams = omp_thread_count();
  // Create cuda streams and handlers
//  cudaStream_t *streams = new cudaStream_t [number_of_streams];
  cudaStream_t stream;
  // Create a handle for CUBLAS
  cublasHandle_t handle;
//  cublasHandle_t *handlers = new cublasHandle_t [number_of_streams];

  // Initialize
//  #pragma omp parallel for num_threads(number_of_streams) shared(streams, handlers)
//  for(size_t i = 0; i < number_of_streams; ++i){
//    cudaStreamCreate(&streams[omp_get_thread_num()]);
//  }

  cublasCreate(&handle);
  cudaStreamCreate(&stream);
  cublasSetStream(handle, stream);

//  #pragma omp parallel for num_threads(number_of_streams)
//  for(size_t i = 0; i < number_of_streams; ++i){
//    cudaStreamCreate(&streams[i]);
//  }

  // Stopping condition in Hogben, L. (Ed.). (2013). Handbook of Linear Algebra (2nd ed.). Chapman and Hall/CRC. https://doi.org/10.1201/b16113
  size_t m_ordering = (n + 1) / 2;
//  std::cout << "m_ordering: " << m_ordering << "\n";

  // Ordering in  A. Sameh. On Jacobi and Jacobi-like algorithms for a parallel computer. Math. Comput., 25:579–590,
  // 1971
  for (size_t k = 1; k < m_ordering; ++k) {
    size_t p = 0;
    size_t p_trans = 0;
    size_t q_trans = 0;
//    #pragma omp parallel for private(p, p_trans, q_trans)
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

//      cudaEvent_t calculateColumnNorms;
//      cudaEventCreate(&calculateColumnNorms);
      double alpha, beta, gamma;
      // \alpha = a_p^T\cdot a_q, \beta = a_p^T\cdot a_p, \gamma = a_q^T\cdot a_q
      cublasDdot(handle, m,
                 reinterpret_cast<const double *>(A.elements + m * p_trans), 1,
                 reinterpret_cast<const double *>(A.elements + m * q_trans), 1,
                 &alpha);
      cublasDdot(handle, m,
                 reinterpret_cast<const double *>(A.elements + m * p_trans), 1,
                 reinterpret_cast<const double *>(A.elements + m * p_trans), 1,
                 &beta);
      cublasDdot(handle, m,
                 reinterpret_cast<const double *>(A.elements + m * q_trans), 1,
                 reinterpret_cast<const double *>(A.elements + m * q_trans), 1,
                 &gamma);
      cudaStreamSynchronize(stream);
//      cudaEventRecord(calculateColumnNorms, stream);

      if (abs(alpha) > tolerance) {
        double tau = (gamma - beta) / (2.0 * alpha);
        double t = 0.0;

        if (tau >= 0) {
          t = 1.0 / (tau + sqrt(1 + (tau * tau)));
        } else {
          t = 1.0 / (tau - sqrt(1 + (tau * tau)));
        }

        const double c_schur = 1.0 / sqrt(1 + (t * t));
        const double s_schur = t * c_schur;

//          cudaStreamWaitEvent(stream, calculateColumnNorms);
        cublasDrot(handle, m,
                   A.elements + m * q_trans, 1,
                   A.elements + m * p_trans, 1,
                   &c_schur, &s_schur);

        cublasDrot(handle, m,
                   V.elements + m * q_trans, 1,
                   V.elements + m * p_trans, 1,
                   &c_schur, &s_schur);
      }
    }

//    #pragma omp parallel for num_threads(number_of_streams)
//    for(size_t i = 0; i < number_of_streams; ++i){
//      cudaStreamSynchronize(streams[i]);
//    }
    cudaStreamSynchronize(stream);
  }

  for (size_t k = m_ordering; k < 2 * m_ordering; ++k) {
    size_t p = 0;
    size_t p_trans = 0;
    size_t q_trans = 0;
//    #pragma omp parallel for private(p, p_trans, q_trans)
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

//      cudaEvent_t calculateColumnNorms;
//      cudaEventCreate(&calculateColumnNorms);
      double alpha = 0.0, beta = 0.0, gamma = 0.0;
      // \alpha = a_p^T\cdot a_q, \beta = a_p^T\cdot a_p, \gamma = a_q^T\cdot a_q
      cublasDdot(handle, m,
                 reinterpret_cast<const double *>(A.elements + m * p_trans), 1,
                 reinterpret_cast<const double *>(A.elements + m * q_trans), 1,
                 &alpha);
      cublasDdot(handle, m,
                 reinterpret_cast<const double *>(A.elements + m * p_trans), 1,
                 reinterpret_cast<const double *>(A.elements + m * p_trans), 1,
                 &beta);
      cublasDdot(handle, m,
                 reinterpret_cast<const double *>(A.elements + m * q_trans), 1,
                 reinterpret_cast<const double *>(A.elements + m * q_trans), 1,
                 &gamma);
      cudaStreamSynchronize(stream);
//      cudaEventRecord(calculateColumnNorms, stream);

      if (abs(alpha) > tolerance) {
        double tau = (gamma - beta) / (2.0 * alpha);
        double t = 0.0;

        if (tau >= 0) {
          t = 1.0 / (tau + sqrt(1 + (tau * tau)));
        } else {
          t = 1.0 / (tau - sqrt(1 + (tau * tau)));
        }

        const double c_schur = 1.0 / sqrt(1 + (t * t));
        const double s_schur = t * c_schur;

//          cudaStreamWaitEvent(stream, calculateColumnNorms);
//        cudaStreamSynchronize(streams[omp_get_thread_num()]);
        cublasDrot(handle, m,
                   A.elements + m * q_trans, 1,
                   A.elements + m * p_trans, 1,
                   &c_schur, &s_schur);

        cublasDrot(handle, m,
                   V.elements + m * q_trans, 1,
                   V.elements + m * p_trans, 1,
                   &c_schur, &s_schur);
      }
    }

//    #pragma omp parallel for num_threads(number_of_streams)
//    for(size_t i = 0; i < number_of_streams; ++i){
//      cudaStreamSynchronize(streams[i]);
//    }
    cudaStreamSynchronize(stream);
  }
  /*
  for (size_t k = 1; k < m_ordering; ++k) {
    size_t p = 0;
    size_t p_trans = 0;
    size_t q_trans = 0;
    #pragma omp parallel for num_threads(number_of_streams) private(p, p_trans, q_trans) shared(handlers, streams)
    for (size_t q = m_ordering - k + 1; q <= n - k; ++q) {
      cublasSetStream(handlers[omp_get_thread_num()], streams[omp_get_thread_num()]);
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

//      cudaEvent_t calculateColumnNorms;
//      cudaEventCreate(&calculateColumnNorms);
      double alpha, beta, gamma;
      // \alpha = a_p^T\cdot a_q, \beta = a_p^T\cdot a_p, \gamma = a_q^T\cdot a_q
      cublasDdot(handle, m,
                 reinterpret_cast<const double *>(A.elements + m * p_trans), 1,
                 reinterpret_cast<const double *>(A.elements + m * q_trans), 1,
                 &alpha);
      cublasDdot(handle, m,
                 reinterpret_cast<const double *>(A.elements + m * p_trans), 1,
                 reinterpret_cast<const double *>(A.elements + m * p_trans), 1,
                 &beta);
      cublasDdot(handle, m,
                 reinterpret_cast<const double *>(A.elements + m * q_trans), 1,
                 reinterpret_cast<const double *>(A.elements + m * q_trans), 1,
                 &gamma);
//      cudaEventRecord(calculateColumnNorms, stream);

      if (abs(alpha) > tolerance) {
        double tau = (gamma - beta) / (2.0 * alpha);
        double t = 0.0;

        if (tau >= 0) {
          t = 1.0 / (tau + sqrt(1 + (tau * tau)));
        } else {
          t = 1.0 / (tau - sqrt(1 + (tau * tau)));
        }

        const double c_schur = 1.0 / sqrt(1 + (t * t));
        const double s_schur = t * c_schur;

//          cudaStreamWaitEvent(stream, calculateColumnNorms);
        cudaStreamSynchronize(streams[omp_get_thread_num()]);
        cublasDrot(handle, m,
                   A.elements + m * q_trans, 1,
                   A.elements + m * p_trans, 1,
                   &c_schur, &s_schur);

        cublasDrot(handle, m,
                   V.elements + m * q_trans, 1,
                   V.elements + m * p_trans, 1,
                   &c_schur, &s_schur);
      }
    }

    #pragma omp parallel for num_threads(number_of_streams) shared(streams)
    for(size_t i = 0; i < number_of_streams; ++i){
      cudaStreamSynchronize(streams[i]);
    }
//    cudaStreamSynchronize(stream);
  }

  for (size_t k = m_ordering; k < 2 * m_ordering; ++k) {
    size_t p = 0;
    size_t p_trans = 0;
    size_t q_trans = 0;
    #pragma omp parallel for num_threads(number_of_streams) private(p, p_trans, q_trans) shared(handlers, streams)
    for (size_t q = (4 * m_ordering) - n - k; q < (3 * m_ordering) - k; ++q) {
      cublasSetStream(handlers[omp_get_thread_num()], streams[omp_get_thread_num()]);
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

//      cudaEvent_t calculateColumnNorms;
//      cudaEventCreate(&calculateColumnNorms);
      double alpha = 0.0, beta = 0.0, gamma = 0.0;
      // \alpha = a_p^T\cdot a_q, \beta = a_p^T\cdot a_p, \gamma = a_q^T\cdot a_q
      cublasDdot(handle, m,
                 reinterpret_cast<const double *>(A.elements + m * p_trans), 1,
                 reinterpret_cast<const double *>(A.elements + m * q_trans), 1,
                 &alpha);
      cublasDdot(handle, m,
                 reinterpret_cast<const double *>(A.elements + m * p_trans), 1,
                 reinterpret_cast<const double *>(A.elements + m * p_trans), 1,
                 &beta);
      cublasDdot(handle, m,
                 reinterpret_cast<const double *>(A.elements + m * q_trans), 1,
                 reinterpret_cast<const double *>(A.elements + m * q_trans), 1,
                 &gamma);
//      cudaEventRecord(calculateColumnNorms, stream);

      if (abs(alpha) > tolerance) {
        double tau = (gamma - beta) / (2.0 * alpha);
        double t = 0.0;

        if (tau >= 0) {
          t = 1.0 / (tau + sqrt(1 + (tau * tau)));
        } else {
          t = 1.0 / (tau - sqrt(1 + (tau * tau)));
        }

        const double c_schur = 1.0 / sqrt(1 + (t * t));
        const double s_schur = t * c_schur;

//          cudaStreamWaitEvent(stream, calculateColumnNorms);
        cudaStreamSynchronize(streams[omp_get_thread_num()]);
//        cudaStreamSynchronize(stream);
        cublasDrot(handle, m,
                   A.elements + m * q_trans, 1,
                   A.elements + m * p_trans, 1,
                   &c_schur, &s_schur);

        cublasDrot(handle, m,
                   V.elements + m * q_trans, 1,
                   V.elements + m * p_trans, 1,
                   &c_schur, &s_schur);
      }
    }

    #pragma omp parallel for num_threads(number_of_streams) shared(streams)
    for(size_t i = 0; i < number_of_streams; ++i){
      cudaStreamSynchronize(streams[i]);
    }
//    cudaStreamSynchronize(stream);
  }
*/
//  std::cout << "Finalized jacobi sweep!!\n";

  // Compute \Sigma
  Matrix s_copy(1, std::min(m, n));
  for (size_t k = 0; k < std::min(m, n); ++k) {
    double result;
    cublasDnrm2(handle, m, reinterpret_cast<const double *>(A.elements + m * k), 1, &result);
    s_copy.elements[k] = result;
  }
  s.copy_from_host(s_copy);
//  std::cout << "Finalized normalization!!\n";

  //Compute U
  U.copy_from_device(A);
  for (size_t i = 0; i < m; ++i) {
    double element = s_copy.elements[i];
    double scale = 1.0 / element;
    cublasDscal(handle, m, reinterpret_cast<const double *>(&scale), U.elements + m * i, 1);
  }
//  std::cout << "Finalized U!!\n";

  // Destroy stream
//  #pragma omp parallel for num_threads(number_of_streams)
//  for(size_t i = 0; i < number_of_streams; ++i){
//    cudaStreamDestroy(streams[i]);
//  }
//  #pragma omp parallel for num_threads(number_of_streams) shared(streams, handlers)
//  for(size_t i = 0; i < number_of_streams; ++i){
//    cudaStreamDestroy(streams[omp_get_thread_num()]);
//    cublasDestroy(handlers[omp_thread_count()]);
//  }
  cudaStreamDestroy(stream);
  // Destroy the handle
  cublasDestroy(handle);
}

void cuda_dgesvd_kernel(SVD_OPTIONS jobu,
                SVD_OPTIONS jobv,
                size_t m,
                size_t n,
                Matrix &A,
                size_t lda,
                Matrix &s,
                Matrix &V,
                size_t ldv) {
  auto num_of_threads = Thesis::omp_thread_count();

  int threadsPerBlock = 16;
  dim3 A_blocksPerGrid  (ceil( float(m) / threadsPerBlock ));
  dim3 V_blocksPerGrid  (ceil( float(n) / threadsPerBlock ));

  // Initializing V = 1
  if (jobv == AllVec) {
    for (size_t i = 0; i < n; ++i) {
      V.elements[iteratorC(i, i, ldv)] = 1.0;
    }
  } else if (jobv == SomeVec) {
    for (size_t i = 0; i < std::min(m, n); ++i) {
      V.elements[iteratorC(i, i, ldv)] = 1.0;
    }
  }

  size_t m_ordering = (n + 1) / 2;

#ifdef DEBUG
  // Report Matrix A^T * A
  std::cout << std::fixed << std::setprecision(3) << "A^T * A: \n";
  for (size_t indexRow = 0; indexRow < m; ++indexRow) {
    for (size_t indexCol = 0; indexCol < n; ++indexCol) {
      double value = 0.0;
      for(size_t k_dot = 0; k_dot < m; ++k_dot){
        value += A.elements[iterator(k_dot, indexRow, lda)] * A.elements[iterator(k_dot, indexCol, lda)];
      }
      std::cout << value << " ";
    }
    std::cout << '\n';
  }
#endif
  // Stopping condition in Hogben, L. (Ed.). (2013). Handbook of Linear Algebra (2nd ed.). Chapman and Hall/CRC. https://doi.org/10.1201/b16113
  size_t maxIterations = 1;

  std::vector<double*> d_p_vectors(num_of_threads);
  std::vector<double*> d_q_vectors(num_of_threads);
  std::vector<double*> d_v_p_vectors(num_of_threads);
  std::vector<double*> d_v_q_vectors(num_of_threads);

  for(size_t i = 0; i < num_of_threads; i++){
    cudaMalloc(&d_p_vectors[i], m * sizeof(double));
    cudaMalloc(&d_q_vectors[i], m * sizeof(double));
    cudaMalloc(&d_v_p_vectors[i], n * sizeof(double));
    cudaMalloc(&d_v_q_vectors[i], n * sizeof(double));
  }

  for(auto number_iterations = 0; number_iterations < maxIterations; ++number_iterations){
    // Ordering in  A. Sameh. On Jacobi and Jacobi-like algorithms for a parallel computer. Math. Comput., 25:579–590,
    // 1971
    for (size_t k = 1; k < m_ordering; ++k) {
      size_t p = 0;
      size_t p_trans = 0;
      size_t q_trans = 0;

      #pragma omp parallel for private(p, p_trans, q_trans)
      for (size_t q = m_ordering - k + 1; q <= n - k; ++q) {
        size_t thread_id = omp_get_thread_num();
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
          tmp_p = A.elements[iteratorC(i, p_trans, lda)];
          tmp_q = A.elements[iteratorC(i, q_trans, lda)];
          alpha += tmp_p * tmp_q;
          beta += tmp_p * tmp_p;
          gamma += tmp_q * tmp_q;
        }

        // Schur
        double c_schur = 1.0, s_schur = 0.0, aqq = gamma, app = beta, apq = alpha;

        if (abs(apq) > tolerance) {
          double tau = (aqq - app) / (2.0 * apq);
          double t = 0.0;

          if (tau >= 0) {
            t = 1.0 / (tau + sqrt(1 + (tau * tau)));
          } else {
            t = 1.0 / (tau - sqrt(1 + (tau * tau)));
          }

          c_schur = 1.0 / sqrt(1 + (t * t));
          s_schur = t * c_schur;

          cudaMemcpy(d_p_vectors[thread_id], (A.elements + p_trans*lda), m * sizeof(double),
                     cudaMemcpyHostToDevice);
          cudaMemcpy(d_q_vectors[thread_id], (A.elements + q_trans*lda), m * sizeof(double),
                     cudaMemcpyHostToDevice);

//          CUDAMatrix d_p_vector(, m), d_q_vector((A.elements + q_trans*lda), m);

          jacobi_rotation<<<A_blocksPerGrid, threadsPerBlock>>>(m, d_p_vectors[thread_id], d_q_vectors[thread_id], c_schur, s_schur);

          cudaMemcpy((A.elements + p_trans*lda), d_p_vectors[thread_id],m * sizeof(double),
                     cudaMemcpyDeviceToHost);
          cudaMemcpy((A.elements + q_trans*lda), d_q_vectors[thread_id],m * sizeof(double),
                     cudaMemcpyDeviceToHost);
//          d_p_vector.copy_to_host((A.elements + p_trans*lda), m);
//          d_q_vector.copy_to_host((A.elements + q_trans*lda), m);
//
//          d_p_vector.free();
//          d_q_vector.free();

//          if (jobv == AllVec || jobv == SomeVec) {
//            CUDAMatrix d_v_p_vector((V.elements + p_trans*lda), n), d_v_q_vector((V.elements + q_trans*lda), n);

          cudaMemcpy(d_v_p_vectors[thread_id], (V.elements + p_trans*ldv), n * sizeof(double),
                     cudaMemcpyHostToDevice);
          cudaMemcpy(d_v_q_vectors[thread_id], (V.elements + q_trans*ldv), n * sizeof(double),
                     cudaMemcpyHostToDevice);
          jacobi_rotation<<<V_blocksPerGrid, threadsPerBlock>>>(n, d_v_p_vectors[thread_id], d_v_q_vectors[thread_id], c_schur, s_schur);

          cudaMemcpy((V.elements + p_trans*ldv), d_v_p_vectors[thread_id],n * sizeof(double),
                     cudaMemcpyDeviceToHost);
          cudaMemcpy((V.elements + q_trans*ldv), d_v_q_vectors[thread_id],n * sizeof(double),
                     cudaMemcpyDeviceToHost);
//            d_v_p_vector.copy_to_host((V.elements + p_trans*lda), n);
//            d_v_q_vector.copy_to_host((V.elements + q_trans*lda), n);

//            d_v_p_vector.free();
//            d_v_q_vector.free();
//          }
        }
      }
    }

    for (size_t k = m_ordering; k < 2 * m_ordering; ++k) {
      size_t thread_id = omp_thread_count();
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
          tmp_p = A.elements[iteratorC(i, p_trans, lda)];
          tmp_q = A.elements[iteratorC(i, q_trans, lda)];
          alpha += tmp_p * tmp_q;
          beta += tmp_p * tmp_p;
          gamma += tmp_q * tmp_q;
        }

        // Schur
        double c_schur = 1.0, s_schur = 0.0, aqq = gamma, app = beta, apq = alpha;

        if (abs(apq) > tolerance) {
          double tau = (aqq - app) / (2.0 * apq);
          double t = 0.0;

          if (tau >= 0) {
            t = 1.0 / (tau + sqrt(1 + (tau * tau)));
          } else {
            t = 1.0 / (tau - sqrt(1 + (tau * tau)));
          }

          c_schur = 1.0 / sqrt(1 + (t * t));
          s_schur = t * c_schur;

          cudaMemcpy(d_p_vectors[thread_id], (A.elements + p_trans*lda), m * sizeof(double),
                     cudaMemcpyHostToDevice);
          cudaMemcpy(d_q_vectors[thread_id], (A.elements + q_trans*lda), m * sizeof(double),
                     cudaMemcpyHostToDevice);

//          CUDAMatrix d_p_vector(, m), d_q_vector((A.elements + q_trans*lda), m);

          jacobi_rotation<<<A_blocksPerGrid, threadsPerBlock>>>(m, d_p_vectors[thread_id], d_q_vectors[thread_id], c_schur, s_schur);

          cudaMemcpy((A.elements + p_trans*lda), d_p_vectors[thread_id],m * sizeof(double),
                     cudaMemcpyDeviceToHost);
          cudaMemcpy((A.elements + q_trans*lda), d_q_vectors[thread_id],m * sizeof(double),
                     cudaMemcpyDeviceToHost);
//          d_p_vector.copy_to_host((A.elements + p_trans*lda), m);
//          d_q_vector.copy_to_host((A.elements + q_trans*lda), m);
//
//          d_p_vector.free();
//          d_q_vector.free();

//          if (jobv == AllVec || jobv == SomeVec) {
//            CUDAMatrix d_v_p_vector((V.elements + p_trans*lda), n), d_v_q_vector((V.elements + q_trans*lda), n);

          cudaMemcpy(d_v_p_vectors[thread_id], (V.elements + p_trans*ldv), n * sizeof(double),
                     cudaMemcpyHostToDevice);
          cudaMemcpy(d_v_q_vectors[thread_id], (V.elements + q_trans*ldv), n * sizeof(double),
                     cudaMemcpyHostToDevice);
          jacobi_rotation<<<V_blocksPerGrid, threadsPerBlock>>>(m, d_v_p_vectors[thread_id], d_v_q_vectors[thread_id], c_schur, s_schur);

          cudaMemcpy((V.elements + p_trans*ldv), d_v_p_vectors[thread_id],n * sizeof(double),
                     cudaMemcpyDeviceToHost);
          cudaMemcpy((V.elements + q_trans*ldv), d_v_q_vectors[thread_id],n * sizeof(double),
                     cudaMemcpyDeviceToHost);
//            d_v_p_vector.copy_to_host((V.elements + p_trans*lda), n);
//            d_v_q_vector.copy_to_host((V.elements + q_trans*lda), n);

//            d_v_p_vector.free();
//            d_v_q_vector.free();
//          }
        }
      }
    }
  }

  for(size_t i = 0; i < num_of_threads; i++){
    cudaFree(d_p_vectors[i]);
    cudaFree(d_q_vectors[i]);
    cudaFree(d_v_p_vectors[i]);
    cudaFree(d_v_q_vectors[i]);
  }

  std::cout << "How many repetitions?: " << maxIterations << "\n";

  // Compute \Sigma
#pragma omp parallel for
  for (size_t k = 0; k < std::min(m, n); ++k) {
    for (size_t i = 0; i < m; ++i) {
      s.elements[k] += A.elements[iteratorC(i, k, lda)] * A.elements[iteratorC(i, k, lda)];
    }
    s.elements[k] = sqrt(s.elements[k]);
  }

  //Compute U
  if (jobu == AllVec) {
#pragma omp parallel for
    for (size_t i = 0; i < m; ++i) {
      for (size_t j = 0; j < m; ++j) {
        A.elements[iteratorC(j, i, lda)] = A.elements[iteratorC(j, i, lda)] / s.elements[i];
      }
    }
  } else if (jobu == SomeVec) {
#pragma omp parallel for
    for (size_t k = 0; k < std::min(m, n); ++k) {
      for (size_t i = 0; i < m; ++i) {
        A.elements[iteratorC(i, k, lda)] = A.elements[iteratorC(i, k, lda)] / s.elements[k];
      }
    }
  }

//  delete []ordering_array;
}

void cuda_dgesvd_kernel(SVD_OPTIONS jobu,
                        SVD_OPTIONS jobv,
                        size_t m,
                        size_t n,
                        CUDAMatrix &A,
                        size_t lda,
                        CUDAMatrix &s,
                        CUDAMatrix &V,
                        size_t ldv) {
  // Create a handle for CUBLAS
  cublasHandle_t handle;
  cublasCreate(&handle);

  int threadsPerBlock = 16;
  dim3 A_blocksPerGrid  (ceil( float(m) / threadsPerBlock ));
  dim3 V_blocksPerGrid  (ceil( float(n) / threadsPerBlock ));

  size_t m_ordering = (n + 1) / 2;

#ifdef DEBUG
  // Report Matrix A^T * A
  std::cout << std::fixed << std::setprecision(3) << "A^T * A: \n";
  for (size_t indexRow = 0; indexRow < m; ++indexRow) {
    for (size_t indexCol = 0; indexCol < n; ++indexCol) {
      double value = 0.0;
      for(size_t k_dot = 0; k_dot < m; ++k_dot){
        value += A.elements[iterator(k_dot, indexRow, lda)] * A.elements[iterator(k_dot, indexCol, lda)];
      }
      std::cout << value << " ";
    }
    std::cout << '\n';
  }
#endif
  // Stopping condition in Hogben, L. (Ed.). (2013). Handbook of Linear Algebra (2nd ed.). Chapman and Hall/CRC. https://doi.org/10.1201/b16113
  size_t maxIterations = 1;

  for(auto number_iterations = 0; number_iterations < maxIterations; ++number_iterations){
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

        cublasDdot(handle, m,
                   reinterpret_cast<const double *>(A.elements + m * p_trans), 1,
                   reinterpret_cast<const double *>(A.elements + m * q_trans), 1,
                   &alpha);
        cublasDdot(handle, m,
                   reinterpret_cast<const double *>(A.elements + m * p_trans), 1,
                   reinterpret_cast<const double *>(A.elements + m * p_trans), 1,
                   &beta);
        cublasDdot(handle, m,
                   reinterpret_cast<const double *>(A.elements + m * q_trans), 1,
                   reinterpret_cast<const double *>(A.elements + m * q_trans), 1,
                   &gamma);

        // Schur
        double c_schur = 1.0, s_schur = 0.0, aqq = gamma, app = beta, apq = alpha;

        if (abs(apq) > tolerance) {
          double tau = (aqq - app) / (2.0 * apq);
          double t = 0.0;

          if (tau >= 0) {
            t = 1.0 / (tau + sqrt(1 + (tau * tau)));
          } else {
            t = 1.0 / (tau - sqrt(1 + (tau * tau)));
          }

          c_schur = 1.0 / sqrt(1 + (t * t));
          s_schur = t * c_schur;

          jacobi_rotation<<<A_blocksPerGrid, threadsPerBlock>>>(m, (A.elements + p_trans*lda), (A.elements + q_trans*lda), c_schur, s_schur);
          jacobi_rotation<<<V_blocksPerGrid, threadsPerBlock>>>(m, (V.elements + p_trans*lda), (V.elements + q_trans*lda), c_schur, s_schur);
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
        cublasDdot(handle, m,
                   reinterpret_cast<const double *>(A.elements + m * p_trans), 1,
                   reinterpret_cast<const double *>(A.elements + m * q_trans), 1,
                   &alpha);
        cublasDdot(handle, m,
                   reinterpret_cast<const double *>(A.elements + m * p_trans), 1,
                   reinterpret_cast<const double *>(A.elements + m * p_trans), 1,
                   &beta);
        cublasDdot(handle, m,
                   reinterpret_cast<const double *>(A.elements + m * q_trans), 1,
                   reinterpret_cast<const double *>(A.elements + m * q_trans), 1,
                   &gamma);

        // Schur
        double c_schur = 1.0, s_schur = 0.0, aqq = gamma, app = beta, apq = alpha;

        if (abs(apq) > tolerance) {
          double tau = (aqq - app) / (2.0 * apq);
          double t = 0.0;

          if (tau >= 0) {
            t = 1.0 / (tau + sqrt(1 + (tau * tau)));
          } else {
            t = 1.0 / (tau - sqrt(1 + (tau * tau)));
          }

          c_schur = 1.0 / sqrt(1 + (t * t));
          s_schur = t * c_schur;

          jacobi_rotation<<<A_blocksPerGrid, threadsPerBlock>>>(m, (A.elements + p_trans*lda), (A.elements + q_trans*lda), c_schur, s_schur);
          jacobi_rotation<<<V_blocksPerGrid, threadsPerBlock>>>(m, (V.elements + p_trans*lda), (V.elements + q_trans*lda), c_schur, s_schur);
        }
      }
    }

#ifdef DEBUG
    // Report Matrix A^T * A
    std::cout << std::fixed << std::setprecision(3) << "A^T * A: \n";
    for (size_t indexRow = 0; indexRow < m; ++indexRow) {
      for (size_t indexCol = 0; indexCol < n; ++indexCol) {
        double value = 0.0;
        for(size_t k_dot = 0; k_dot < m; ++k_dot){
          value += A.elements[iterator(k_dot, indexRow, lda)] * A.elements[iterator(k_dot, indexCol, lda)];
        }
        std::cout << value << " ";
      }
      std::cout << '\n';
    }
#endif
  }

  std::cout << "How many repetitions?: " << maxIterations << "\n";

// Compute \Sigma
//  cudaDeviceSynchronize();
  Matrix s_copy(1, std::min(m, n));
  for (size_t k = 0; k < std::min(m, n); ++k) {
    double result;
    cublasDnrm2(handle, m, reinterpret_cast<const double *>(A.elements + m * k), 1, &result);
    s_copy.elements[k] = result;
  }
  s.copy_from_host(s_copy);
//  std::cout << "Finalized normalization!!\n";

  //Compute U
//  cudaDeviceSynchronize();
  for (size_t i = 0; i < m; ++i) {
    double element = s_copy.elements[i];
    double scale = 1.0 / element;
    cublasDscal(handle, m, reinterpret_cast<const double *>(&scale), A.elements + m * i, 1);
  }

//  delete []ordering_array;
// Destroy the handle
  cublasDestroy(handle);
}

//#ifdef ERASE
void cuda_dgesvd_kernel_streams(SVD_OPTIONS jobu,
                                SVD_OPTIONS jobv,
                                size_t m,
                                size_t n,
                                Matrix &A,
                                size_t lda,
                                Matrix &s,
                                Matrix &V,
                                size_t ldv) {

  size_t m_bytes = m * sizeof(double);
  size_t n_bytes = n * sizeof(double);

  auto num_of_threads = omp_thread_count();

  int threadsPerBlock = 16;
  dim3 A_blocksPerGrid  (ceil( float(m) / threadsPerBlock ));
  dim3 V_blocksPerGrid  (ceil( float(n) / threadsPerBlock ));

  // Initializing V = 1
  if (jobv == AllVec) {
    for (size_t i = 0; i < n; ++i) {
      V.elements[iteratorC(i, i, ldv)] = 1.0;
    }
  } else if (jobv == SomeVec) {
    for (size_t i = 0; i < std::min(m, n); ++i) {
      V.elements[iteratorC(i, i, ldv)] = 1.0;
    }
  }

  size_t m_ordering = (n + 1) / 2;

#ifdef DEBUG
  // Report Matrix A^T * A
  std::cout << std::fixed << std::setprecision(3) << "A^T * A: \n";
  for (size_t indexRow = 0; indexRow < m; ++indexRow) {
    for (size_t indexCol = 0; indexCol < n; ++indexCol) {
      double value = 0.0;
      for(size_t k_dot = 0; k_dot < m; ++k_dot){
        value += A.elements[iterator(k_dot, indexRow, lda)] * A.elements[iterator(k_dot, indexCol, lda)];
      }
      std::cout << value << " ";
    }
    std::cout << '\n';
  }
#endif
  // Stopping condition in Hogben, L. (Ed.). (2013). Handbook of Linear Algebra (2nd ed.). Chapman and Hall/CRC. https://doi.org/10.1201/b16113
  size_t maxIterations = 1;

  for(auto number_iterations = 0; number_iterations < maxIterations; ++number_iterations){
    // Ordering in  A. Sameh. On Jacobi and Jacobi-like algorithms for a parallel computer. Math. Comput., 25:579–590,
    // 1971
    for (size_t k = 1; k < m_ordering; ++k) {
      // Create two stream for A and V
      std::vector<cudaStream_t> streams(2);
      for(auto &stream: streams){
        CHECK_CUDA(cudaStreamCreate(&stream));
      }

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
        double tmp_p, tmp_q;
        for (size_t i = 0; i < m; ++i) {
          tmp_p = A.elements[iteratorC(i, p_trans, lda)];
          tmp_q = A.elements[iteratorC(i, q_trans, lda)];
          alpha += tmp_p * tmp_q;
          beta += tmp_p * tmp_p;
          gamma += tmp_q * tmp_q;
        }

        // Schur
        double c_schur = 1.0, s_schur = 0.0, aqq = 0.0, app = 0.0, apq = alpha;

        // Calculate a_{pp}, a_{qq}, a_{pq}
        for (size_t i = 0; i < m; ++i) {
          double value_p = A.elements[iteratorC(i, p_trans, lda)];
          double value_q = A.elements[iteratorC(i, q_trans, lda)];
          app += value_p * value_p;
          aqq += value_q * value_q;
        }

        if (abs(apq) > tolerance) {
          double tau = (aqq - app) / (2.0 * apq);
          double t = 0.0;

          if (tau >= 0) {
            t = 1.0 / (tau + sqrt(1 + (tau * tau)));
          } else {
            t = 1.0 / (tau - sqrt(1 + (tau * tau)));
          }

          c_schur = 1.0 / sqrt(1 + (t * t));
          s_schur = t * c_schur;

          /* ---------------------------------- Create events --------------------------------- */

//          cudaEvent_t eventMallocA, eventMallocV, eventMemCpyAH2D, eventMemCpyVH2D, eventKernelA, eventKernelV,
//              eventMemCpyAD2H, eventMemCpyVD2H;
//
//          CHECK_CUDA(cudaEventCreate(&eventMallocA));
//          cudaEventCreate(&eventMallocV);
//          cudaEventCreate(&eventMemCpyAH2D);
//          cudaEventCreate(&eventMemCpyVH2D);
//          cudaEventCreate(&eventKernelA);
//          cudaEventCreate(&eventKernelV);
//          cudaEventCreate(&eventMemCpyAD2H);
//          cudaEventCreate(&eventMemCpyVD2H);

          /* ---------------------------------- Malloc and memcpy from host to device --------------------------------- */
          double *d_p_vector, *d_q_vector, *d_v_p_vector, *d_v_q_vector;

          CHECK_CUDA(cudaMallocAsync(&d_p_vector, m_bytes, streams[0]));
          CHECK_CUDA(cudaMallocAsync(&d_q_vector, m_bytes, streams[0]));
//          cudaEventRecord(eventMallocA, stream[0]);

          CHECK_CUDA(cudaMallocAsync(&d_v_p_vector, n_bytes, streams[1]));
          CHECK_CUDA(cudaMallocAsync(&d_v_q_vector, n_bytes, streams[1]));
//          cudaEventRecord(eventMallocV, stream[1]);

//          cudaStreamWaitEvent(stream[0], eventMallocA);
          cudaMemcpyAsync(d_p_vector, &A.elements[iteratorC(0, p_trans, lda)], m * sizeof(double),
                          cudaMemcpyHostToDevice, streams[0]);
          cudaMemcpyAsync(d_q_vector, &A.elements[iteratorC(0, q_trans, lda)], m * sizeof(double),
                          cudaMemcpyHostToDevice, streams[0]);
//          cudaEventRecord(eventMemCpyAH2D, stream[0]);

//          cudaStreamWaitEvent(stream[1], eventMallocV);
          cudaMemcpyAsync(d_v_p_vector, &V.elements[iteratorC(0, p_trans, ldv)], n * sizeof(double),
                          cudaMemcpyHostToDevice, streams[1]);
          cudaMemcpyAsync(d_v_q_vector, &V.elements[iteratorC(0, q_trans, ldv)], n * sizeof(double),
                          cudaMemcpyHostToDevice, streams[1]);
//          cudaEventRecord(eventMemCpyVH2D, stream[1]);

          /* ---------------------------------- Kernel execution --------------------------------- */
//          cudaStreamWaitEvent(stream[0], eventMemCpyAH2D);
          jacobi_rotation<<<A_blocksPerGrid, threadsPerBlock, 0, streams[0]>>>(m, d_p_vector, d_q_vector, c_schur, s_schur);
//          cudaEventRecord(eventKernelA, stream[0]);

//          cudaStreamWaitEvent(stream[1], eventMemCpyVH2D);
          jacobi_rotation<<<V_blocksPerGrid, threadsPerBlock, 0, streams[1]>>>(m, d_v_p_vector, d_v_q_vector, c_schur, s_schur);
//          cudaEventRecord(eventKernelV, stream[1]);

          /* ---------------------------------- Malloc and memcpy from device to host --------------------------------- */
//          cudaStreamWaitEvent(stream[0], eventKernelA);
          cudaMemcpyAsync(&A.elements[iteratorC(0, p_trans, lda)], d_p_vector, m * sizeof(double), cudaMemcpyDeviceToHost, streams[0]);
          cudaMemcpyAsync(&A.elements[iteratorC(0, q_trans, lda)], d_q_vector, m * sizeof(double), cudaMemcpyDeviceToHost, streams[0]);
//          cudaEventRecord(eventMemCpyAD2H, stream[0]);

//          cudaStreamWaitEvent(stream[1], eventKernelV);
          cudaMemcpyAsync(&V.elements[iteratorC(0, p_trans, ldv)], d_v_p_vector, n * sizeof(double), cudaMemcpyDeviceToHost, streams[1]);
          cudaMemcpyAsync(&V.elements[iteratorC(0, q_trans, ldv)], d_v_q_vector, n * sizeof(double), cudaMemcpyDeviceToHost, streams[1]);
//          cudaEventRecord(eventMemCpyVD2H, stream[1]);

          /* ---------------------------------- Free cuda malloc --------------------------------- */
//          cudaStreamWaitEvent(stream[0], eventMemCpyAD2H);
          cudaFreeAsync(d_p_vector, streams[0]);
          cudaFreeAsync(d_q_vector, streams[0]);

//          cudaStreamWaitEvent(stream[1], eventMemCpyVD2H);
          cudaFreeAsync(d_v_p_vector, streams[1]);
          cudaFreeAsync(d_v_q_vector, streams[1]);

          cudaStreamSynchronize(streams[0]);
          cudaStreamSynchronize(streams[1]);

//          cudaEventDestroy(eventMallocA);
//          cudaEventDestroy(eventMallocV);
//          cudaEventDestroy(eventMemCpyAH2D);
//          cudaEventDestroy(eventMemCpyVH2D);
//          cudaEventDestroy(eventKernelA);
//          cudaEventDestroy(eventKernelV);
//          cudaEventDestroy(eventMemCpyAD2H);
//          cudaEventDestroy(eventMemCpyVD2H);

//          for (int i = 0; i < 2; ++i) {
//            cudaStreamDestroy(stream[i]);
//          }
        }

#ifdef DEBUG
        // Report Matrix A^T * A
        std::cout << std::fixed << std::setprecision(3) << "A^T * A: \n";
        for (size_t indexRow = 0; indexRow < m; ++indexRow) {
          for (size_t indexCol = 0; indexCol < n; ++indexCol) {
            double value = 0.0;
            for(size_t k_dot = 0; k_dot < m; ++k_dot){
              value += A.elements[iterator(k_dot, indexRow, lda)] * A.elements[iterator(k_dot, indexCol, lda)];
            }
            std::cout << value << " ";
          }
          std::cout << '\n';
        }
#endif
      }

//      cudaDeviceSynchronize();
      for(auto &stream: streams){
        CHECK_CUDA(cudaStreamDestroy(stream));
      }
    }

    for (size_t k = m_ordering; k < 2 * m_ordering; ++k) {
      // Create two stream for A and V
      std::vector<cudaStream_t> streams(2);
      for(auto &stream: streams){
        CHECK_CUDA(cudaStreamCreate(&stream));
      }

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
        double tmp_p, tmp_q;
        for (size_t i = 0; i < m; ++i) {
          tmp_p = A.elements[iteratorC(i, p_trans, lda)];
          tmp_q = A.elements[iteratorC(i, q_trans, lda)];
          alpha += tmp_p * tmp_q;
          beta += tmp_p * tmp_p;
          gamma += tmp_q * tmp_q;
        }

        // Schur
        double c_schur = 1.0, s_schur = 0.0, aqq = 0.0, app = 0.0, apq = alpha;

        // Calculate a_{pp}, a_{qq}, a_{pq}
        for (size_t i = 0; i < m; ++i) {
          double value_p = A.elements[iteratorC(i, p_trans, lda)];
          double value_q = A.elements[iteratorC(i, q_trans, lda)];
          app += value_p * value_p;
          aqq += value_q * value_q;
        }

        if (abs(apq) > tolerance) {


          double tau = (aqq - app) / (2.0 * apq);
          double t = 0.0;

          if (tau >= 0) {
            t = 1.0 / (tau + sqrt(1 + (tau * tau)));
          } else {
            t = 1.0 / (tau - sqrt(1 + (tau * tau)));
          }

          c_schur = 1.0 / sqrt(1 + (t * t));
          s_schur = t * c_schur;
          /* ---------------------------------- Create events --------------------------------- */

//          cudaEvent_t eventMallocA, eventMallocV, eventMemCpyAH2D, eventMemCpyVH2D, eventKernelA, eventKernelV,
//              eventMemCpyAD2H, eventMemCpyVD2H;
//
//          CHECK_CUDA(cudaEventCreate(&eventMallocA));
//          cudaEventCreate(&eventMallocV);
//          cudaEventCreate(&eventMemCpyAH2D);
//          cudaEventCreate(&eventMemCpyVH2D);
//          cudaEventCreate(&eventKernelA);
//          cudaEventCreate(&eventKernelV);
//          cudaEventCreate(&eventMemCpyAD2H);
//          cudaEventCreate(&eventMemCpyVD2H);

          /* ---------------------------------- Malloc and memcpy from host to device --------------------------------- */
          double *d_p_vector, *d_q_vector, *d_v_p_vector, *d_v_q_vector;

          CHECK_CUDA(cudaMallocAsync(&d_p_vector, m_bytes, streams[0]));
          CHECK_CUDA(cudaMallocAsync(&d_q_vector, m_bytes, streams[0]));
//          cudaEventRecord(eventMallocA, stream[0]);

          CHECK_CUDA(cudaMallocAsync(&d_v_p_vector, n_bytes, streams[1]));
          CHECK_CUDA(cudaMallocAsync(&d_v_q_vector, n_bytes, streams[1]));
//          cudaEventRecord(eventMallocV, stream[1]);

//          cudaStreamWaitEvent(stream[0], eventMallocA);
          cudaMemcpyAsync(d_p_vector, &A.elements[iteratorC(0, p_trans, lda)], m * sizeof(double),
                     cudaMemcpyHostToDevice, streams[0]);
          cudaMemcpyAsync(d_q_vector, &A.elements[iteratorC(0, q_trans, lda)], m * sizeof(double),
                     cudaMemcpyHostToDevice, streams[0]);
//          cudaEventRecord(eventMemCpyAH2D, stream[0]);

//          cudaStreamWaitEvent(stream[1], eventMallocV);
          cudaMemcpyAsync(d_v_p_vector, &V.elements[iteratorC(0, p_trans, ldv)], n * sizeof(double),
                          cudaMemcpyHostToDevice, streams[1]);
          cudaMemcpyAsync(d_v_q_vector, &V.elements[iteratorC(0, q_trans, ldv)], n * sizeof(double),
                          cudaMemcpyHostToDevice, streams[1]);
//          cudaEventRecord(eventMemCpyVH2D, stream[1]);

          /* ---------------------------------- Kernel execution --------------------------------- */
//          cudaStreamWaitEvent(stream[0], eventMemCpyAH2D);
          jacobi_rotation<<<A_blocksPerGrid, threadsPerBlock, 0, streams[0]>>>(m, d_p_vector, d_q_vector, c_schur, s_schur);
//          cudaEventRecord(eventKernelA, stream[0]);

//          cudaStreamWaitEvent(stream[1], eventMemCpyVH2D);
          jacobi_rotation<<<V_blocksPerGrid, threadsPerBlock, 0, streams[1]>>>(m, d_v_p_vector, d_v_q_vector, c_schur, s_schur);
//          cudaEventRecord(eventKernelV, stream[1]);

          /* ---------------------------------- Malloc and memcpy from device to host --------------------------------- */
//          cudaStreamWaitEvent(stream[0], eventKernelA);
          cudaMemcpyAsync(&A.elements[iteratorC(0, p_trans, lda)], d_p_vector, m * sizeof(double), cudaMemcpyDeviceToHost, streams[0]);
          cudaMemcpyAsync(&A.elements[iteratorC(0, q_trans, lda)], d_q_vector, m * sizeof(double), cudaMemcpyDeviceToHost, streams[0]);
//          cudaEventRecord(eventMemCpyAD2H, stream[0]);

//          cudaStreamWaitEvent(stream[1], eventKernelV);
          cudaMemcpyAsync(&V.elements[iteratorC(0, p_trans, ldv)], d_v_p_vector, n * sizeof(double), cudaMemcpyDeviceToHost, streams[1]);
          cudaMemcpyAsync(&V.elements[iteratorC(0, q_trans, ldv)], d_v_q_vector, n * sizeof(double), cudaMemcpyDeviceToHost, streams[1]);
//          cudaEventRecord(eventMemCpyVD2H, stream[1]);

          /* ---------------------------------- Free cuda malloc --------------------------------- */
//          cudaStreamWaitEvent(stream[0], eventMemCpyAD2H);
          cudaFreeAsync(d_p_vector, streams[0]);
          cudaFreeAsync(d_q_vector, streams[0]);

//          cudaStreamWaitEvent(stream[1], eventMemCpyVD2H);
          cudaFreeAsync(d_v_p_vector, streams[1]);
          cudaFreeAsync(d_v_q_vector, streams[1]);

          cudaStreamSynchronize(streams[0]);
          cudaStreamSynchronize(streams[1]);

//          cudaEventDestroy(eventMallocA);
//          cudaEventDestroy(eventMallocV);
//          cudaEventDestroy(eventMemCpyAH2D);
//          cudaEventDestroy(eventMemCpyVH2D);
//          cudaEventDestroy(eventKernelA);
//          cudaEventDestroy(eventKernelV);
//          cudaEventDestroy(eventMemCpyAD2H);
//          cudaEventDestroy(eventMemCpyVD2H);

//          for (int i = 0; i < 2; ++i) {
//            cudaStreamDestroy(stream[i]);
//          }
        }

#ifdef DEBUG
        // Report Matrix A^T * A
        std::cout << std::fixed << std::setprecision(3) << "A^T * A: \n";
        for (size_t indexRow = 0; indexRow < m; ++indexRow) {
          for (size_t indexCol = 0; indexCol < n; ++indexCol) {
            double value = 0.0;
            for(size_t k_dot = 0; k_dot < m; ++k_dot){
              value += A.elements[iterator(k_dot, indexRow, lda)] * A.elements[iterator(k_dot, indexCol, lda)];
            }
            std::cout << value << " ";
          }
          std::cout << '\n';
        }
#endif
      }

      for(auto &stream: streams){
        CHECK_CUDA(cudaStreamDestroy(stream));
      }
    }

#ifdef DEBUG
    // Report Matrix A^T * A
    std::cout << std::fixed << std::setprecision(3) << "A^T * A: \n";
    for (size_t indexRow = 0; indexRow < m; ++indexRow) {
      for (size_t indexCol = 0; indexCol < n; ++indexCol) {
        double value = 0.0;
        for(size_t k_dot = 0; k_dot < m; ++k_dot){
          value += A.elements[iterator(k_dot, indexRow, lda)] * A.elements[iterator(k_dot, indexCol, lda)];
        }
        std::cout << value << " ";
      }
      std::cout << '\n';
    }
#endif
  }

  std::cout << "How many repetitions?: " << maxIterations << "\n";

  // Compute \Sigma
#pragma omp parallel for
  for (size_t k = 0; k < std::min(m, n); ++k) {
    for (size_t i = 0; i < m; ++i) {
      s.elements[k] += A.elements[iteratorC(i, k, lda)] * A.elements[iteratorC(i, k, lda)];
    }
    s.elements[k] = sqrt(s.elements[k]);
  }

  //Compute U
  if (jobu == AllVec) {
#pragma omp parallel for
    for (size_t i = 0; i < m; ++i) {
      for (size_t j = 0; j < m; ++j) {
        A.elements[iteratorC(j, i, m)] = A.elements[iteratorC(j, i, m)] / s.elements[i];
      }
    }
  } else if (jobu == SomeVec) {
#pragma omp parallel for
    for (size_t k = 0; k < std::min(m, n); ++k) {
      for (size_t i = 0; i < m; ++i) {
        A.elements[iteratorC(i, k, m)] = A.elements[iteratorC(i, k, m)] / s.elements[k];
      }
    }
  }

//  delete []ordering_array;
}
//#endif
/***************************************************************************
    Purpose
    -------
    Applies the Jacobi rotation cosine and sine to both arrays x and y.
    In the following way:

    x[i] = c * x[i] - s * y[i]
    y[i] = s * x[i] + c * y[i]
    Arguments
    ---------
    @param[in]
    n       int
            number of elements in array to apply the rotation.

    @param[in,out]
    x       DOUBLE PRECISION array dimension at least n
            The x array to be overwritten.

    @param[in,out]
    y       DOUBLE PRECISION array dimension at least n
            The y array to be overwritten.

    @param[in]
    c       DOUBLE
            Cosine of Jacobi rotation.

    @param[in]
    s       DOUBLE PRECISION
            Sine of Jacobi rotation.
*********************************************************************************/
 __global__ void jacobi_rotation(unsigned int n, double *x, double *y, double c, double s) {
     unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

     if (i < n) {
         double tmp = x[i];
         x[i] = c * tmp - s * y[i];
         y[i] = s * tmp + c * y[i];
     }
 }
}
