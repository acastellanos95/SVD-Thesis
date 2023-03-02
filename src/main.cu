#include <iostream>
#include <random>
#include <omp.h>
#include "../lib/HouseholderMethods.cuh"
#include "../lib/global.cuh"
#include "../lib/Matrix.cuh"

int main() {
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

    const unsigned long A_height = A.height, A_width = A.width;

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
#endif

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
  }
  file.close();

  return 0;
}
