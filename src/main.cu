#include <iostream>
#include <random>
#include <omp.h>
#include <cmath>
#include <unordered_map>
#include <sstream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "../lib/HouseholderMethods.cuh"
#include "../lib/JacobiMethods.cuh"
#include "../lib/global.cuh"
#include "../lib/Matrix.cuh"
#include "../tests/Tests.cuh"
#include <mkl/mkl.h>

int main() {
  // SEED!!!
  const unsigned seed = 1000000;

#ifdef TESTS
//  Thesis::max_iterations_error();
//  Thesis::compare_cuda_operations();
#endif

  size_t begin = 5000;
  size_t end = 10000;
  size_t delta = 1000;
  auto t = std::time(nullptr);
  auto tm = *std::localtime(&t);

  std::ostringstream oss;
  oss << std::put_time(&tm, "%d-%m-%Y-%H-%M-%S");
  auto now_time = oss.str();

  std::stringstream file_output;
  file_output << "NT: " << NT << '\n';
  file_output << "Number of threads: " << omp_get_num_threads() << '\n';
  for (; begin <= end; begin += delta) {
#ifdef SEQUENTIAL
    {
      /* -------------------------------- Test 1 (Squared matrix SVD) sequential -------------------------------- */
      file
          << "-------------------------------- Test 1 (Squared matrix SVD) sequential --------------------------------\n";
      std::cout
          << "-------------------------------- Test 1 (Squared matrix SVD) sequential --------------------------------\n";

      const size_t height = begin;
      const size_t width = begin;

      file_output << "Dimensions, height: " << height << ", width: " << width << "\n";
      std::cout << "Dimensions, height: " << height << ", width: " << width << "\n";
      // Build matrix A and R
      Matrix A, U, V, s, A_copy;
      A.height = height;
      A.width = width;
      A_copy.height = height;
      A_copy.width = width;
      U.height = A.height;
      U.width = A.height;
      V.height = A.width;
      V.width = A.width;
      s.width = std::min(A.height, A.width);

      A.elements = new double[A.height * A.width];
      A_copy.elements = new double[A_copy.height * A_copy.width];
      U.elements = new double[U.height * U.width];
      V.elements = new double[V.height * V.width];
      s.elements = new double[s.width];

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

      // Create a random bidiagonal matrix
      std::default_random_engine e(seed);
      std::uniform_real_distribution<double> uniform_dist(1.0, 2.0);
      for (size_t indexRow = 0; indexRow < std::min<size_t>(A_height, A_width); ++indexRow) {
        double value = uniform_dist(e);
        A.elements[Thesis::IteratorC(indexRow, indexRow, A_height)] = value;
        A_copy.elements[Thesis::IteratorC(indexRow, indexRow, A_height)] = value;
      }

      for (size_t indexRow = 0; indexRow < (std::min<size_t>(A_height, A_width) - 1); ++indexRow) {
        double value = uniform_dist(e);
        A.elements[Thesis::IteratorC(indexRow, indexRow + 1, A_height)] = value;
        A_copy.elements[Thesis::IteratorC(indexRow, indexRow + 1, A_height)] = value;
      }

#ifdef REPORT
      // Report Matrix A
    file_output << std::fixed << std::setprecision(3) << "A: \n";
    std::cout << std::fixed << std::setprecision(3) << "A: \n";
    for (size_t indexRow = 0; indexRow < A_height; ++indexRow) {
      for (size_t indexCol = 0; indexCol < A_width; ++indexCol) {
        file_output << A.elements[Thesis::IteratorC(indexRow, indexCol, A_height)] << " ";
        std::cout << A.elements[Thesis::IteratorC(indexRow, indexCol, A_height)] << " ";
      }
      file_output << '\n';
      std::cout << '\n';
    }
    // Report Matrix A^T * A
//    std::cout << std::fixed << std::setprecision(3) << "A^T * A: \n";
//    for (size_t indexRow = 0; indexRow < A.width; ++indexRow) {
//      for (size_t indexCol = 0; indexCol < A.width; ++indexCol) {
//        double value = 0.0;
//        for(size_t k_dot = 0; k_dot < A.height; ++k_dot){
//          value += A.elements[Thesis::IteratorC(k_dot, indexRow, A.height)] * A.elements[Thesis::IteratorC(k_dot, indexCol, A.height)];
//        }
//        std::cout << value << " ";
//      }
//      std::cout << '\n';
//    }
#endif

      // Calculate SVD decomposition
      double ti = omp_get_wtime();
      Thesis::sequential_dgesvd(Thesis::AllVec,
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
      double tf = omp_get_wtime();
      double time = tf - ti;

      file_output << "SVD sequential time with U,V calculation: " << time << "\n";
      std::cout << "SVD sequential time with U,V calculation: " << time << "\n";

      double maxError = 0.0;
      for (size_t indexRow = 0; indexRow < A_height; ++indexRow) {
        for (size_t indexCol = 0; indexCol < A_width; ++indexCol) {
          double value = 0.0;
          for(size_t k_dot = 0; k_dot < A_width; ++k_dot){
            value += U.elements[Thesis::IteratorC(indexRow, k_dot, A_height)] * s.elements[k_dot] * V.elements[Thesis::IteratorC(indexCol, k_dot, A_height)];
          }
          double diff = std::abs(A_copy.elements[Thesis::IteratorC(indexRow, indexCol, A_height)] - value);
          maxError = std::max<double>(maxError, diff);
        }
      }

      file_output << "max error between A and USV: " << maxError << "\n";
      std::cout << "max error between A and USV: " << maxError << "\n";


#ifdef REPORT
      // Report Matrix A=USV
    std::cout << std::fixed << std::setprecision(3) << "A=USV^T: \n";
    for (size_t indexRow = 0; indexRow < A_height; ++indexRow) {
      for (size_t indexCol = 0; indexCol < A_width; ++indexCol) {
        double value = 0.0;
        for(size_t k_dot = 0; k_dot < A_width; ++k_dot){
          value += U.elements[Thesis::IteratorC(indexRow, k_dot, A_height)] * s.elements[k_dot] * V.elements[Thesis::IteratorC(indexCol, k_dot, A_height)];
        }
        A_tmp.elements[Thesis::IteratorC(indexRow, indexCol, A_height)] = value;
        std::cout << value << " ";
      }
      std::cout << '\n';
    }

    // Report Matrix U
    file_output << std::fixed << std::setprecision(3) << "U: \n";
    std::cout << std::fixed << std::setprecision(3) << "U: \n";
    for (size_t indexRow = 0; indexRow < A_height; ++indexRow) {
      for (size_t indexCol = 0; indexCol < A_height; ++indexCol) {
        file_output << U.elements[Thesis::IteratorC(indexRow, indexCol, A_height)] << " ";
        std::cout << U.elements[Thesis::IteratorC(indexRow, indexCol, A_height)] << " ";
      }
      file_output << '\n';
      std::cout << '\n';
    }

    // Report \Sigma
    file_output << std::fixed << std::setprecision(3) << "sigma: \n";
    std::cout << std::fixed << std::setprecision(3) << "sigma: \n";
    for (size_t indexCol = 0; indexCol < std::min(A_height, A_width); ++indexCol) {
      file_output << s.elements[indexCol] << " ";
      std::cout << s.elements[indexCol] << " ";
    }
    file_output << '\n';
    std::cout << '\n';

    // Report Matrix V
    file_output << std::fixed << std::setprecision(3) << "V: \n";
    std::cout << std::fixed << std::setprecision(3) << "V: \n";
    for (size_t indexRow = 0; indexRow < A_width; ++indexRow) {
      for (size_t indexCol = 0; indexCol < A_width; ++indexCol) {
        file_output << V.elements[Thesis::IteratorC(indexRow, indexCol, A_width)] << " ";
        std::cout << V.elements[Thesis::IteratorC(indexRow, indexCol, A_width)] << " ";
      }
      file_output << '\n';
      std::cout << '\n';
    }
#endif
    }
#endif

#ifdef OMP
    {
      double time_avg = 0.0;
      for(auto i_repeat = 0; i_repeat < 10; ++i_repeat){
        {
          // Build matrix A and R
          /* -------------------------------- Test 1 (Squared matrix SVD) OMP -------------------------------- */
          file_output
              << "-------------------------------- Test 1 (Squared matrix SVD) OMP --------------------------------\n";
          std::cout
              << "-------------------------------- Test 1 (Squared matrix SVD) OMP --------------------------------\n";

          const size_t height = begin;
          const size_t width = begin;

          file_output << "Dimensions, height: " << height << ", width: " << width << "\n";
          std::cout << "Dimensions, height: " << height << ", width: " << width << "\n";

          Matrix A(height, width), U(height, height), V(width, width), s(1, std::min(A.height, A.width)), A_copy(height, width);

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
          std::default_random_engine e(seed);
          std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);
          for (size_t indexRow = 0; indexRow < std::min<size_t>(A_height, A_width); ++indexRow) {
            double value = uniform_dist(e);
            A.elements[iterator(indexRow, indexRow, A_height)] = value;
            A_copy.elements[iterator(indexRow, indexRow, A_height)] = value;
          }

          for (size_t indexRow = 0; indexRow < (std::min<size_t>(A_height, A_width) - 1); ++indexRow) {
            double value = uniform_dist(e);
            A.elements[iterator(indexRow, indexRow + 1, A_height)] = value;
            A_copy.elements[iterator(indexRow, indexRow + 1, A_height)] = value;
          }

#ifdef REPORT
          // Report Matrix A
  file_output << std::fixed << std::setprecision(3) << "A: \n";
  std::cout << std::fixed << std::setprecision(3) << "A: \n";
  for (size_t indexRow = 0; indexRow < A_height; ++indexRow) {
    for (size_t indexCol = 0; indexCol < A_width; ++indexCol) {
      file_output << A.elements[Thesis::IteratorC(indexRow, indexCol, A_height)] << " ";
      std::cout << A.elements[Thesis::IteratorC(indexRow, indexCol, A_height)] << " ";
    }
    file_output << '\n';
    std::cout << '\n';
  }
  // Report Matrix A^T * A
//    std::cout << std::fixed << std::setprecision(3) << "A^T * A: \n";
//    for (size_t indexRow = 0; indexRow < A.width; ++indexRow) {
//      for (size_t indexCol = 0; indexCol < A.width; ++indexCol) {
//        double value = 0.0;
//        for(size_t k_dot = 0; k_dot < A.height; ++k_dot){
//          value += A.elements[Thesis::IteratorC(k_dot, indexRow, A.height)] * A.elements[Thesis::IteratorC(k_dot, indexCol, A.height)];
//        }
//        std::cout << value << " ";
//      }
//      std::cout << '\n';
//    }
#endif

          // Calculate SVD decomposition
          double ti = omp_get_wtime();
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
          double tf = omp_get_wtime();
          double time = tf - ti;
          time_avg += time;

          file_output << "SVD OMP time with U,V calculation: " << time << "\n";
          std::cout << "SVD OMP time with U,V calculation: " << time << "\n";

//          double maxError = 0.0;
//          for (size_t indexRow = 0; indexRow < A_height; ++indexRow) {
//            for (size_t indexCol = 0; indexCol < A_width; ++indexCol) {
//              double value = 0.0;
//              for (size_t k_dot = 0; k_dot < A_width; ++k_dot) {
//                value += U.elements[iterator(indexRow, k_dot, A_height)] * s.elements[k_dot]
//                    * V.elements[iterator(indexCol, k_dot, A_height)];
//              }
//              double diff = std::abs(A_copy.elements[iterator(indexRow, indexCol, A_height)] - value);
//              maxError = std::max<double>(maxError, diff);
//            }
//          }
//
//          file_output << "max error between A and USV: " << maxError << "\n";
//          std::cout << "max error between A and USV: " << maxError << "\n";

#ifdef REPORT
          // Report Matrix A=USV
  std::cout << std::fixed << std::setprecision(3) << "A=USV^T: \n";
  for (size_t indexRow = 0; indexRow < A_height; ++indexRow) {
    for (size_t indexCol = 0; indexCol < A_width; ++indexCol) {
      double value = 0.0;
      for(size_t k_dot = 0; k_dot < A_width; ++k_dot){
        value += U.elements[Thesis::IteratorC(indexRow, k_dot, A_height)] * s.elements[k_dot] * V.elements[Thesis::IteratorC(indexCol, k_dot, A_height)];
      }
      std::cout << value << " ";
    }
    std::cout << '\n';
  }

  // Report Matrix U
  file_output << std::fixed << std::setprecision(3) << "U: \n";
  std::cout << std::fixed << std::setprecision(3) << "U: \n";
  for (size_t indexRow = 0; indexRow < A_height; ++indexRow) {
    for (size_t indexCol = 0; indexCol < A_height; ++indexCol) {
      file_output << U.elements[Thesis::IteratorC(indexRow, indexCol, A_height)] << " ";
      std::cout << U.elements[Thesis::IteratorC(indexRow, indexCol, A_height)] << " ";
    }
    file_output << '\n';
    std::cout << '\n';
  }

  // Report \Sigma
  file_output << std::fixed << std::setprecision(3) << "sigma: \n";
  std::cout << std::fixed << std::setprecision(3) << "sigma: \n";
  for (size_t indexCol = 0; indexCol < std::min(A_height, A_width); ++indexCol) {
    file_output << s.elements[indexCol] << " ";
    std::cout << s.elements[indexCol] << " ";
  }
  file_output << '\n';
  std::cout << '\n';

  // Report Matrix V
  file_output << std::fixed << std::setprecision(3) << "V: \n";
  std::cout << std::fixed << std::setprecision(3) << "V: \n";
  for (size_t indexRow = 0; indexRow < A_width; ++indexRow) {
    for (size_t indexCol = 0; indexCol < A_width; ++indexCol) {
      file_output << V.elements[Thesis::IteratorC(indexRow, indexCol, A_width)] << " ";
      std::cout << V.elements[Thesis::IteratorC(indexRow, indexCol, A_width)] << " ";
    }
    file_output << '\n';
    std::cout << '\n';
  }
#endif
        }
      }

      std::cout << "Tiempo promedio: " << (time_avg / 10.0) << "\n";
    }
#endif

#ifdef CUDA
    {
      double time_avg = 0.0;
      for(auto i_repeat = 0; i_repeat < 10; ++i_repeat){
        {
          // Build matrix A and R
          /* -------------------------------- Test 1 (Squared matrix SVD) CUDA -------------------------------- */
          file_output
              << "-------------------------------- Test 1 (Squared matrix SVD) CUDA --------------------------------\n";
          std::cout
              << "-------------------------------- Test 1 (Squared matrix SVD) CUDA --------------------------------\n";

          const size_t height = begin;
          const size_t width = begin;

          file_output << "Dimensions, height: " << height << ", width: " << width << "\n";
          std::cout << "Dimensions, height: " << height << ", width: " << width << "\n";

          Matrix A(height, width), U(height, height), V(width, width), s(1, std::min(A.height, A.width)), A_copy(height, width);

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
//      std::default_random_engine e(seed);
//      std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);
//      for (size_t indexRow = 0; indexRow < std::min<size_t>(A_height, A_width); ++indexRow) {
//        double value = uniform_dist(e);
//        A.elements[iterator(indexRow, indexRow, A_height)] = value;
//        A_copy.elements[iterator(indexRow, indexRow, A_height)] = value;
//      }
//
//      for (size_t indexRow = 0; indexRow < (std::min<size_t>(A_height, A_width) - 1); ++indexRow) {
//        double value = uniform_dist(e);
//        A.elements[iterator(indexRow, indexRow + 1, A_height)] = value;
//        A_copy.elements[iterator(indexRow, indexRow + 1, A_height)] = value;
//      }

          // Create R matrix
          std::default_random_engine e(seed);
          std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);
          for (size_t indexRow = 0; indexRow < std::min<size_t>(A_height, A_width); ++indexRow) {
            for (size_t indexCol = indexRow; indexCol < std::min<size_t>(A_height, A_width); ++indexCol) {
              double value = uniform_dist(e);
              A.elements[iterator(indexRow, indexCol, A_height)] = value;
              A_copy.elements[iterator(indexRow, indexCol, A_height)] = value;
            }
          }

          for (size_t i = 0; i < A.width; ++i) {
            V.elements[iterator(i, i, A_width)] = 1.0;
          }

          CUDAMatrix d_A( A), d_s(s), d_U(U), d_V(V);
          std::cout << "Initialized!!\n";

#ifdef REPORT
          // Report Matrix A
  file_output << std::fixed << std::setprecision(3) << "A: \n";
  std::cout << std::fixed << std::setprecision(3) << "A: \n";
  for (size_t indexRow = 0; indexRow < A_height; ++indexRow) {
    for (size_t indexCol = 0; indexCol < A_width; ++indexCol) {
      file_output << A.elements[Thesis::IteratorC(indexRow, indexCol, A_height)] << " ";
      std::cout << A.elements[Thesis::IteratorC(indexRow, indexCol, A_height)] << " ";
    }
    file_output << '\n';
    std::cout << '\n';
  }
  // Report Matrix A^T * A
//    std::cout << std::fixed << std::setprecision(3) << "A^T * A: \n";
//    for (size_t indexRow = 0; indexRow < A.width; ++indexRow) {
//      for (size_t indexCol = 0; indexCol < A.width; ++indexCol) {
//        double value = 0.0;
//        for(size_t k_dot = 0; k_dot < A.height; ++k_dot){
//          value += A.elements[Thesis::IteratorC(k_dot, indexRow, A.height)] * A.elements[Thesis::IteratorC(k_dot, indexCol, A.height)];
//        }
//        std::cout << value << " ";
//      }
//      std::cout << '\n';
//    }
#endif

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
          time_avg += time;

          file_output << "SVD CUDA time with U,V calculation: " << time << "\n";
          std::cout << "SVD CUDA time with U,V calculation: " << time << "\n";

          d_A.copy_to_host(A), d_s.copy_to_host(s), d_U.copy_to_host(U), d_V.copy_to_host(V);

//          double maxError = 0.0;
//          for (size_t indexRow = 0; indexRow < A_height; ++indexRow) {
//            for (size_t indexCol = 0; indexCol < A_width; ++indexCol) {
//              double value = 0.0;
//              for (size_t k_dot = 0; k_dot < A_width; ++k_dot) {
//                value += U.elements[iterator(indexRow, k_dot, A_height)] * s.elements[k_dot]
//                    * V.elements[iterator(indexCol, k_dot, A_height)];
//              }
//              double diff = std::abs(A_copy.elements[iterator(indexRow, indexCol, A_height)] - value);
//              maxError = std::max<double>(maxError, diff);
//            }
//          }
//
//          file_output << "max error between A and USV: " << maxError << "\n";
//          std::cout << "max error between A and USV: " << maxError << "\n";

#ifdef REPORT
          // Report Matrix A=USV
  std::cout << std::fixed << std::setprecision(3) << "A=USV^T: \n";
  for (size_t indexRow = 0; indexRow < A_height; ++indexRow) {
    for (size_t indexCol = 0; indexCol < A_width; ++indexCol) {
      double value = 0.0;
      for(size_t k_dot = 0; k_dot < A_width; ++k_dot){
        value += U.elements[Thesis::IteratorC(indexRow, k_dot, A_height)] * s.elements[k_dot] * V.elements[Thesis::IteratorC(indexCol, k_dot, A_height)];
      }
      A_tmp.elements[Thesis::IteratorC(indexRow, indexCol, A_height)] = value;
      std::cout << value << " ";
    }
    std::cout << '\n';
  }

  // Report Matrix U
  file_output << std::fixed << std::setprecision(3) << "U: \n";
  std::cout << std::fixed << std::setprecision(3) << "U: \n";
  for (size_t indexRow = 0; indexRow < A_height; ++indexRow) {
    for (size_t indexCol = 0; indexCol < A_height; ++indexCol) {
      file_output << U.elements[Thesis::IteratorC(indexRow, indexCol, A_height)] << " ";
      std::cout << U.elements[Thesis::IteratorC(indexRow, indexCol, A_height)] << " ";
    }
    file_output << '\n';
    std::cout << '\n';
  }

  // Report \Sigma
  file_output << std::fixed << std::setprecision(3) << "sigma: \n";
  std::cout << std::fixed << std::setprecision(3) << "sigma: \n";
  for (size_t indexCol = 0; indexCol < std::min(A_height, A_width); ++indexCol) {
    file_output << s.elements[indexCol] << " ";
    std::cout << s.elements[indexCol] << " ";
  }
  file_output << '\n';
  std::cout << '\n';

  // Report Matrix V
  file_output << std::fixed << std::setprecision(3) << "V: \n";
  std::cout << std::fixed << std::setprecision(3) << "V: \n";
  for (size_t indexRow = 0; indexRow < A_width; ++indexRow) {
    for (size_t indexCol = 0; indexCol < A_width; ++indexCol) {
      file_output << V.elements[Thesis::IteratorC(indexRow, indexCol, A_width)] << " ";
      std::cout << V.elements[Thesis::IteratorC(indexRow, indexCol, A_width)] << " ";
    }
    file_output << '\n';
    std::cout << '\n';
  }
#endif
          d_A.free(), d_s.free(), d_U.free(), d_V.free();
        }
      }

      std::cout << "Tiempo promedio: " << (time_avg / 10.0) << "\n";
    }
#endif

#ifdef IMKL
    {
      double time_avg = 0.0;
      for(auto i_repeat = 0; i_repeat < 10; ++i_repeat){
        {
          // Build matrix A and R
          /* -------------------------------- Test 1 (Squared matrix SVD) MKL Computes the singular value decomposition of a real matrix using Jacobi plane rotations. -------------------------------- */
          file_output
              << "-------------------------------- Test 1 (Squared matrix SVD) MKL Computes the singular value decomposition of a real matrix using Jacobi plane rotations. --------------------------------\n";
          std::cout
              << "-------------------------------- Test 1 (Squared matrix SVD) MKL Computes the singular value decomposition of a real matrix using Jacobi plane rotations. --------------------------------\n";

          const size_t height = begin;
          const size_t width = begin;

          file_output << "Dimensions, height: " << height << ", width: " << width << "\n";
          std::cout << "Dimensions, height: " << height << ", width: " << width << "\n";

          Matrix A(height, width), U(height, height), V(width, width), s(1, std::min(A.height, A.width)), A_copy(height, width), stat(1,6);

          const unsigned long A_height = A.height, A_width = A.width;

          std::fill_n(U.elements, U.height * U.width, 0.0);
          std::fill_n(V.elements, V.height * V.width, 0.0);
          std::fill_n(A.elements, A.height * A.width, 0.0);
          std::fill_n(A_copy.elements, A_copy.height * A_copy.width, 0.0);

          // Select iterator
          auto iterator = Thesis::IteratorC;

          // Create R matrix
          std::default_random_engine e(seed);
          std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);
          for (size_t indexRow = 0; indexRow < std::min<size_t>(A_height, A_width); ++indexRow) {
            for (size_t indexCol = indexRow; indexCol < std::min<size_t>(A_height, A_width); ++indexCol) {
              double value = uniform_dist(e);
              A.elements[iterator(indexRow, indexCol, A_height)] = value;
              A_copy.elements[iterator(indexRow, indexCol, A_height)] = value;
            }
          }

//            for (size_t i = 0; i < A.width; ++i) {
//              V.elements[iterator(i, i, A_width)] = 1.0;
//            }

#ifdef REPORT
          // Report Matrix A
file_output << std::fixed << std::setprecision(3) << "A: \n";
std::cout << std::fixed << std::setprecision(3) << "A: \n";
for (size_t indexRow = 0; indexRow < A_height; ++indexRow) {
  for (size_t indexCol = 0; indexCol < A_width; ++indexCol) {
    file_output << A.elements[Thesis::IteratorC(indexRow, indexCol, A_height)] << " ";
    std::cout << A.elements[Thesis::IteratorC(indexRow, indexCol, A_height)] << " ";
  }
  file_output << '\n';
  std::cout << '\n';
}
// Report Matrix A^T * A
//    std::cout << std::fixed << std::setprecision(3) << "A^T * A: \n";
//    for (size_t indexRow = 0; indexRow < A.width; ++indexRow) {
//      for (size_t indexCol = 0; indexCol < A.width; ++indexCol) {
//        double value = 0.0;
//        for(size_t k_dot = 0; k_dot < A.height; ++k_dot){
//          value += A.elements[Thesis::IteratorC(k_dot, indexRow, A.height)] * A.elements[Thesis::IteratorC(k_dot, indexCol, A.height)];
//        }
//        std::cout << value << " ";
//      }
//      std::cout << '\n';
//    }
#endif

          // Calculate SVD decomposition
          lapack_int info;
          double ti = omp_get_wtime();
          info = LAPACKE_dgesvj(LAPACK_COL_MAJOR, 'U', 'U', 'V', height, width, A.elements, height, s.elements, width, V.elements, width , stat.elements);
          double tf = omp_get_wtime();
          double time = tf - ti;
          time_avg += time;

          if(info > 0){
            file_output << "SVD don't converge\n";
            std::cout << "SVD don't converge\n";
          } else if(info == 0){
            file_output << "SVD converge\n";
            std::cout << "SVD converge\n";
          } else {
            file_output << "SVD error\n";
            std::cout << "SVD error\n";
          }

          file_output << "scale: " << stat.elements[0] << "\n";
          std::cout << "scale: " << stat.elements[0] << "\n";

          file_output << "SVD OMP time with U,V calculation: " << time << "\n";
          std::cout << "SVD OMP time with U,V calculation: " << time << "\n";

//          double maxError = 0.0;
//          for (size_t indexRow = 0; indexRow < A_height; ++indexRow) {
//            for (size_t indexCol = 0; indexCol < A_width; ++indexCol) {
//              double value = 0.0;
//              for (size_t k_dot = 0; k_dot < A_width; ++k_dot) {
//                value += A.elements[iterator(indexRow, k_dot, A_height)] * (stat.elements[0] * s.elements[k_dot])
//                    * V.elements[iterator(indexCol, k_dot, A_height)];
//              }
//              double diff = std::abs(A_copy.elements[iterator(indexRow, indexCol, A_height)] - value);
//              maxError = std::max<double>(maxError, diff);
//            }
//          }
//
//          file_output << "max error between A and USV: " << maxError << "\n";
//          std::cout << "max error between A and USV: " << maxError << "\n";

#ifdef REPORT
          // Report Matrix A=USV
std::cout << std::fixed << std::setprecision(3) << "A=USV^T: \n";
for (size_t indexRow = 0; indexRow < A_height; ++indexRow) {
  for (size_t indexCol = 0; indexCol < A_width; ++indexCol) {
    double value = 0.0;
    for(size_t k_dot = 0; k_dot < A_width; ++k_dot){
      value += U.elements[Thesis::IteratorC(indexRow, k_dot, A_height)] * s.elements[k_dot] * V.elements[Thesis::IteratorC(indexCol, k_dot, A_height)];
    }
    std::cout << value << " ";
  }
  std::cout << '\n';
}

// Report Matrix U
file_output << std::fixed << std::setprecision(3) << "U: \n";
std::cout << std::fixed << std::setprecision(3) << "U: \n";
for (size_t indexRow = 0; indexRow < A_height; ++indexRow) {
  for (size_t indexCol = 0; indexCol < A_height; ++indexCol) {
    file_output << U.elements[Thesis::IteratorC(indexRow, indexCol, A_height)] << " ";
    std::cout << U.elements[Thesis::IteratorC(indexRow, indexCol, A_height)] << " ";
  }
  file_output << '\n';
  std::cout << '\n';
}

// Report \Sigma
file_output << std::fixed << std::setprecision(3) << "sigma: \n";
std::cout << std::fixed << std::setprecision(3) << "sigma: \n";
for (size_t indexCol = 0; indexCol < std::min(A_height, A_width); ++indexCol) {
  file_output << s.elements[indexCol] << " ";
  std::cout << s.elements[indexCol] << " ";
}
file_output << '\n';
std::cout << '\n';

// Report Matrix V
file_output << std::fixed << std::setprecision(3) << "V: \n";
std::cout << std::fixed << std::setprecision(3) << "V: \n";
for (size_t indexRow = 0; indexRow < A_width; ++indexRow) {
  for (size_t indexCol = 0; indexCol < A_width; ++indexCol) {
    file_output << V.elements[Thesis::IteratorC(indexRow, indexCol, A_width)] << " ";
    std::cout << V.elements[Thesis::IteratorC(indexRow, indexCol, A_width)] << " ";
  }
  file_output << '\n';
  std::cout << '\n';
}
#endif
        }
      }

      std::cout << "Tiempo promedio: " << (time_avg / 10.0) << "\n";
    }
#endif
  }

  std::ofstream file("reporte-" + now_time + ".txt", std::ofstream::out | std::ofstream::trunc);
  file << file_output.rdbuf();
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
    Matrix A(height, width), R(height, width), Q(height, height);

    std::fill_n(Q.elements, Q.height * Q.width, 0.0);

    for (auto index = 0; index < Q.height; ++index) {
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
//    QRDecompositionParallelWithB(A, Q, R, file);
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