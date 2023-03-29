#include <iostream>
#include <random>
#include <omp.h>
#include <cmath>
#include "../lib/HouseholderMethods.cuh"
#include "../lib/JacobiMethods.cuh"
#include "../lib/global.cuh"
#include "../lib/Matrix.cuh"

int main(){
  // SEED!!!
  const unsigned seed = 1000000;

  #ifdef TESTS
  {
  std::cout << "Schur Elimination\n";

  Matrix A;
  A.height = 15;
  A.width = 10;
  A.elements = new double[A.height * A.width];

  std::cout << "Dimensions, height: " << A.height << ", width: " << A.width << "\n";

  std::default_random_engine e(seed);
  std::uniform_real_distribution<double> uniform_dist(1.0, 2.0);
  for (size_t indexRow = 0; indexRow < A.height; ++indexRow) {
    for (size_t indexCol = 0; indexCol < A.width; ++indexCol) {
      double value = uniform_dist(e);
      A.elements[Thesis::IteratorC(indexRow, indexCol, A.height)] = value;
    }
  }

  // Report Matrix A
  std::cout << std::fixed << std::setprecision(3) << "A: \n";
  for (size_t indexRow = 0; indexRow < A.height; ++indexRow) {
    for (size_t indexCol = 0; indexCol < A.width; ++indexCol) {
      std::cout << A.elements[Thesis::IteratorC(indexRow, indexCol, A.height)] << " ";
    }
    std::cout << '\n';
  }

  // Report Matrix A^T * A
  std::cout << std::fixed << std::setprecision(3) << "A^T * A: \n";
  for (size_t indexRow = 0; indexRow < A.width; ++indexRow) {
    for (size_t indexCol = 0; indexCol < A.width; ++indexCol) {
      double value = 0.0;
      for(size_t k_dot = 0; k_dot < A.height; ++k_dot){
        value += A.elements[Thesis::IteratorC(k_dot, indexRow, A.height)] * A.elements[Thesis::IteratorC(k_dot, indexCol, A.height)];
      }
      std::cout << value << " ";
    }
    std::cout << '\n';
  }

  size_t p = 0;
  size_t q = 5;

  auto [c_schur, s_schur] = Thesis::non_sym_Schur_non_ordered(Thesis::IteratorC, A.height, A.width, A, A.height, p, q);

  for(size_t i = 0; i < A.height; ++i){
    auto tmp_p = c_schur *  A.elements[Thesis::IteratorC(i, p, A.height)] - s_schur * A.elements[Thesis::IteratorC(i, q, A.height)];
    auto tmp_q = s_schur *  A.elements[Thesis::IteratorC(i, p, A.height)] + c_schur * A.elements[Thesis::IteratorC(i, q, A.height)];
    A.elements[Thesis::IteratorC(i, p, A.height)] = tmp_p;
    A.elements[Thesis::IteratorC(i, q, A.height)] = tmp_q;
  }

  // Report Matrix A^T * A
  std::cout << std::fixed << std::setprecision(3) << "A^T * A: \n";
  for (size_t indexRow = 0; indexRow < A.width; ++indexRow) {
    for (size_t indexCol = 0; indexCol < A.width; ++indexCol) {
      double value = 0.0;
      for(size_t k_dot = 0; k_dot < A.height; ++k_dot){
        value += A.elements[Thesis::IteratorC(k_dot, indexRow, A.height)] * A.elements[Thesis::IteratorC(k_dot, indexCol, A.height)];
      }
      std::cout << value << " ";
    }
    std::cout << '\n';
  }
  std::cout << '\n';
}
  #endif

  size_t begin = 10;
  size_t end = 30;
  size_t delta = 10;
  auto t = std::time(nullptr);
  auto tm = *std::localtime(&t);

  std::ostringstream oss;
  oss << std::put_time(&tm, "%d-%m-%Y-%H-%M-%S");
  auto now_time = oss.str();

  std::ofstream file("reporte-" + now_time + ".txt", std::ofstream::out | std::ofstream::trunc);
  file << "NT: " << NT << '\n';
  file << "Number of threads: " << omp_get_num_threads() << '\n';
  for (; begin <= end; begin += delta) {
    /* -------------------------------- Test 1 (Squared matrix SVD) sequential -------------------------------- */
    file
        << "-------------------------------- Test 1 (Squared matrix SVD) sequential --------------------------------\n";
    std::cout
        << "-------------------------------- Test 1 (Squared matrix SVD) sequential --------------------------------\n";

    const size_t height = begin;
    const size_t width = begin;

    file << "Dimensions, height: " << height << ", width: " << width << "\n";
    std::cout << "Dimensions, height: " << height << ", width: " << width << "\n";

    #ifdef SEQUENTIAL
    // Build matrix A and R
    Matrix A, U, V, s;
    A.height = height;
    A.width = width;
    U.height = A.height;
    U.width = A.height;
    V.height = A.width;
    V.width = A.width;
    s.width = std::min(A.height, A.width);

    A.elements = new double[A.height * A.width];
    U.elements = new double[U.height * U.width];
    V.elements = new double[V.height * V.width];
    s.elements = new double[s.width];

    const unsigned long A_height = A.height, A_width = A.width;

    std::fill_n(U.elements, U.height * U.width, 0.0);
    std::fill_n(V.elements, V.height * V.width, 0.0);
    std::fill_n(A.elements, A.height * A.width, 0.0);

    // Create a random matrix
//    std::default_random_engine e(seed);
//    std::uniform_real_distribution<double> uniform_dist(1.0, 2.0);
//    for (size_t indexRow = 0; indexRow < A_height; ++indexRow) {
//      for (size_t indexCol = 0; indexCol < A_width; ++indexCol) {
//        double value = uniform_dist(e);
//        A.elements[Thesis::IteratorC(indexRow, indexCol, A_height)] = value;
//      }
//    }

    // Create a random bidiagonal matrix
    std::default_random_engine e(seed);
    std::uniform_real_distribution<double> uniform_dist(1.0, 2.0);
    for (size_t indexRow = 0; indexRow < std::min<size_t>(A_height, A_width); ++indexRow) {
      double value = uniform_dist(e);
      A.elements[Thesis::IteratorC(indexRow, indexRow, A_height)] = value;
    }

    for (size_t indexRow = 0; indexRow < (std::min<size_t>(A_height, A_width) - 1); ++indexRow) {
      double value = uniform_dist(e);
      A.elements[Thesis::IteratorC(indexRow, indexRow + 1, A_height)] = value;
    }

    #ifdef REPORT
    // Report Matrix A
    file << std::fixed << std::setprecision(3) << "A: \n";
    std::cout << std::fixed << std::setprecision(3) << "A: \n";
    for (size_t indexRow = 0; indexRow < A_height; ++indexRow) {
      for (size_t indexCol = 0; indexCol < A_width; ++indexCol) {
        file << A.elements[Thesis::IteratorC(indexRow, indexCol, A_height)] << " ";
        std::cout << A.elements[Thesis::IteratorC(indexRow, indexCol, A_height)] << " ";
      }
      file << '\n';
      std::cout << '\n';
    }
    // Report Matrix A^T * A
    std::cout << std::fixed << std::setprecision(3) << "A^T * A: \n";
    for (size_t indexRow = 0; indexRow < A.width; ++indexRow) {
      for (size_t indexCol = 0; indexCol < A.width; ++indexCol) {
        double value = 0.0;
        for(size_t k_dot = 0; k_dot < A.height; ++k_dot){
          value += A.elements[Thesis::IteratorC(k_dot, indexRow, A.height)] * A.elements[Thesis::IteratorC(k_dot, indexCol, A.height)];
        }
        std::cout << value << " ";
      }
      std::cout << '\n';
    }
    #endif

    // Calculate SVD decomposition
    double ti = omp_get_wtime();
    Thesis::sequential_dgesvd(Thesis::AllVec, Thesis::AllVec, A.height, A.width, Thesis::COL_MAJOR, A, A_height, s, U, A_height, V, A_width);
    double tf = omp_get_wtime();
    double time = tf - ti;

    file << "SVD sequential time with U,V calculation: " << time << "\n";
    std::cout << "SVD sequential time with U,V calculation: " << time << "\n";

    #ifdef REPORT
    // Report Matrix A=USV
    std::cout << std::fixed << std::setprecision(3) << "A=USV^T: \n";
    for (size_t indexRow = 0; indexRow < A_height; ++indexRow) {
      for (size_t indexCol = 0; indexCol < A_width; ++indexCol) {
        double value = 0.0;
        for(size_t k_dot = 0; k_dot < A_width; ++k_dot){
          value += U.elements[Thesis::IteratorC(indexRow, k_dot, A_height)] * s.elements[k_dot] * V.elements[Thesis::IteratorC(indexCol, k_dot, A_height)];
        }
//        A_tmp.elements[Thesis::IteratorC(indexRow, indexCol, A_height)] = value;
        std::cout << value << " ";
      }
      std::cout << '\n';
    }

    // Report Matrix U
    file << std::fixed << std::setprecision(3) << "U: \n";
    std::cout << std::fixed << std::setprecision(3) << "U: \n";
    for (size_t indexRow = 0; indexRow < A_height; ++indexRow) {
      for (size_t indexCol = 0; indexCol < A_height; ++indexCol) {
        file << U.elements[Thesis::IteratorC(indexRow, indexCol, A_height)] << " ";
        std::cout << U.elements[Thesis::IteratorC(indexRow, indexCol, A_height)] << " ";
      }
      file << '\n';
      std::cout << '\n';
    }

    // Report \Sigma
    file << std::fixed << std::setprecision(3) << "sigma: \n";
    std::cout << std::fixed << std::setprecision(3) << "sigma: \n";
    for (size_t indexCol = 0; indexCol < std::min(A_height, A_width); ++indexCol) {
      file << s.elements[indexCol] << " ";
      std::cout << s.elements[indexCol] << " ";
    }
    file << '\n';
    std::cout << '\n';

    // Report Matrix V
    file << std::fixed << std::setprecision(3) << "V: \n";
    std::cout << std::fixed << std::setprecision(3) << "V: \n";
    for (size_t indexRow = 0; indexRow < A_width; ++indexRow) {
      for (size_t indexCol = 0; indexCol < A_width; ++indexCol) {
        file << V.elements[Thesis::IteratorC(indexRow, indexCol, A_width)] << " ";
        std::cout << V.elements[Thesis::IteratorC(indexRow, indexCol, A_width)] << " ";
      }
      file << '\n';
      std::cout << '\n';
    }
    #endif
    #endif

  }

  file.close();
  return 0;
}

int main1() {
  // SEED!!!
  const unsigned seed = 1000000;
  size_t begin = 1000;
  size_t end = 2000;
  size_t delta = 1000;
  auto t = std::time(nullptr);
  auto tm = *std::localtime(&t);

  std::ostringstream oss;
  oss << std::put_time(&tm, "%d-%m-%Y-%H-%M-%S");
  auto now_time = oss.str();

  std::ofstream file("reporte-" + now_time + ".txt", std::ofstream::out | std::ofstream::trunc);
  file << "NT: " << NT << '\n';
  file << "Number of threads: " << omp_get_num_threads() << '\n';
  for (; begin <= end; begin += delta) {
    /* -------------------------------- Test 1 (Squared matrix QR decomposition) CUDA -------------------------------- */

    file
        << "-------------------------------- Test 1 (Squared matrix QR decomposition) OMP --------------------------------\n";
    std::cout
        << "-------------------------------- Test 1 (Squared matrix QR decomposition) OMP --------------------------------\n";

    const size_t height = begin;
    const size_t width = begin;

    file << "Dimensions, height: " << height << ", width: " << width << "\n";
    std::cout << "Dimensions, height: " << height << ", width: " << width << "\n";

    const unsigned long A_height = height, A_width = width;
#ifdef OMP
    // Build matrix A and R
    Matrix A, R, Q;
    A.height = R.height = height;
    A.width = R.width = width;
    Q.height = A.height;
    Q.width = A.height;

    A.elements = new double[A.height * A.width];
    Q.elements = new double[Q.height * Q.width];
    R.elements = new double[R.height * R.width];

    std::fill_n(Q.elements, Q.height * Q.width, 0.0);

    for(auto index = 0; index < Q.height; ++index){
      Q.elements[index * Q.width + index] = 1.0;
    }

    std::default_random_engine e(seed);
    std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);
    for (size_t indexRow = 0; indexRow < A_height; ++indexRow) {
      for (size_t indexCol = 0; indexCol < A_width; ++indexCol) {
        double value = uniform_dist(e);
        A.elements[indexRow * A_width + indexCol] = value;
      }
    }

#ifdef REPORT
    // Report Matrix A
    file << std::fixed << std::setprecision(3) << "A: \n";
    std::cout << std::fixed << std::setprecision(3) << "A: \n";
    for (size_t indexRow = 0; indexRow < A_height; ++indexRow) {
      for (size_t indexCol = 0; indexCol < A_width; ++indexCol) {
        file << A.elements[indexRow * A_width + indexCol] << " ";
        std::cout << A.elements[indexRow * A_width + indexCol] << " ";
      }
      file << '\n';
      std::cout << '\n';
    }
#endif

    // Calculate QR decomposition
    double ti = omp_get_wtime();
    QRDecompositionParallelWithB(A, Q, R, file);
    double tf = omp_get_wtime();
    double time = tf - ti;

    file << "QR decomposition OMP time with Q calculation: " << time << "\n";
    std::cout << "QR decomposition OMP time with Q calculation: " << time << "\n";

#ifdef DEBUG
    // Biggest error in comparison
    double maxError = 0.0;
    for (size_t rowIndex = 0; rowIndex < A_height; ++rowIndex) {
      for (size_t colIndex = rowIndex; colIndex < A_width; ++colIndex) {
        double tmp = 0.0;
        for(size_t k = 0; k < A_height; ++k){
          tmp += Q.elements[k*Q.width + rowIndex] * R.elements[k*R.width + colIndex];
        }
        maxError = std::max<double>(std::abs(A.elements[rowIndex * A_width + colIndex] - tmp), maxError);
      }
    }

    file << "Biggest error between A and QR: " << std::to_string(maxError) << "\n";
    std::cout << "Biggest error between A and QR: " << std::to_string(maxError) << "\n";
#endif

#ifdef REPORT
    // Report Matrix R
    file << std::fixed << std::setprecision(3) << "R: \n";
    std::cout << std::fixed << std::setprecision(3) << "R: \n";
    for (size_t indexRow = 0; indexRow < A_height; ++indexRow) {
      for (size_t indexCol = 0; indexCol < A_width; ++indexCol) {
        file << R.elements[indexRow * A_width + indexCol] << " ";
        std::cout << R.elements[indexRow * A_width + indexCol] << " ";
      }
      file << '\n';
      std::cout << '\n';
    }
#endif
#endif
  }
  file.close();

  return 0;
}