//
// Created by andre on 4/18/23.
//

#ifndef SVD_THESIS_TESTS_TESTS_CUH_
#define SVD_THESIS_TESTS_TESTS_CUH_

#include "../lib/JacobiMethods.cuh"
#include "../lib/Utils.cuh"
#include "../lib/Matrix.cuh"
#include <random>
#include <cublas_v2.h>

namespace Thesis {
  void max_iterations_error();
  void compare_cuda_operations();
  void compare_times_dot_product();
  void compare_times_jacobi_matrix_product();
}

#endif //SVD_THESIS_TESTS_TESTS_CUH_
