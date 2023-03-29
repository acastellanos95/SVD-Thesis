//
// Created by andre on 2/03/23.
//

#include "HouseholderMethods.cuh"

/**
 * @brief QR decomposition using householder triangulation. B and R must be initialized before using this function.
 *
 * @param A Matrix to be decomposed of m x n dimension
 * @param B Matrix that holds Householder multiplicacion transformations of the form H_1H_2H_3...B of m x m dimension
 * @param R Matrix upper triangular of m x n dimension
 */
void Thesis::QRDecompositionParallelWithB(const Matrix &A, Matrix &B, Matrix &R, std::ofstream &file) {

#ifdef DEBUG
  if (A.height >= A.width) {
    std::cout << "A podría tener columnas linealmente independientes\n";
  } else {
    std::cout << "A tiene columnas linealmente dependientes\n";
  }

  if (B.height != B.width || B.height != A.height) {
    throw std::runtime_error("Q tiene dimensiones erroneas");
  } else if (R.width != A.width || R.height != A.height) {
    throw std::runtime_error("R tiene dimensiones erroneas");
  }
#endif

  // -------------------------------------------------------------------------------------------------------------------
  // Copy A into R
  // region
  memcpy(R.elements, A.elements, A.width * A.height * sizeof(double));
  // endregion

  // Constants
  const unsigned long A_width = A.width, A_height = A.height;

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

// ---------------------------------------------------------------------------------------------------------------------
// --------------------------------------------Householder triangulation------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------------

  // region Initialize number of iterations, u, v, holding variables and norm variables
  auto min = std::min(A.width, A.height);
  // u, v vectors and to be defined values
  auto *u = new double[A_height];
  auto *v = new double[A_width + A_height];
  double mu2, beta1, norm_column, norm_u_squared;
  // endregion

  for (size_t index_column = 0; index_column < min; ++index_column) {
#ifdef REPORT
    file
        << "-------------------------------- index_column " + std::to_string(index_column) + " --------------------------------\n";
    std::cout
        << "-------------------------------- index_column " + std::to_string(index_column) + " --------------------------------\n";
#endif

    norm_u_squared = 0.0;
    norm_column = 0.0;
    mu2 = 0.0;

// ---------------------------------------------------------------------------------------------------------------------
// 1. Copy elements of column vector of A at index_column to u omitting the first one
// 2. Compute norm squared of u omitting the first one
// ---------------------------------------------------------------------------------------------------------------------
    for (size_t index_of_u = index_column + 1; index_of_u < A_height; ++index_of_u) {
      double u_i = u[index_of_u] = R.elements[index_of_u * A_width + index_column];
      mu2 += u_i * u_i;
    }

#ifdef REPORT
    // Report u[index_column + 1,..,m]
    file << "1. ----------- report u[index_column + 1,..,m]\n";
    std::cout << "1. ----------- report u[index_column + 1,..,m]\n";
    file << std::fixed << "||u[index_column + 1,..,m]||_2^2: " << mu2 << "\n";
    std::cout << std::fixed << "||u[index_column + 1,..,m]||_2^2: " << mu2 << "\n";
    file << std::fixed << std::setprecision(3) << "u[index_column + 1,..,m]: \n";
    std::cout << std::fixed << std::setprecision(3) << "u[index_column + 1,..,m]: \n";
    for (size_t index_of_u = index_column + 1; index_of_u < A_height; ++index_of_u) {
      file << u[index_of_u] << " ";
      std::cout << u[index_of_u] << " ";
    }
    file << '\n';
    std::cout << '\n';
#endif
// ---------------------------------------------------------------------------------------------------------------------
// 3. Compute norm of the vector column of A at index_column
// 4. Compute the first element in u that inserts 0 where needed
// 5. Compute the squared norm of u
// 6. Compute the scalar \beta = 2 / ||u||_2^2
// ---------------------------------------------------------------------------------------------------------------------

    // region
    // Get a_i column norm
    double x_0 = R.elements[index_column * A_width + index_column];
    norm_column = x_0 * x_0 + mu2;
    norm_column = sqrt(norm_column);

    // Get u
    double sign_x_0 = x_0 >= 0.0 ? 1.0 : -1.0;
    double u_0 = u[index_column] = x_0 + (sign_x_0 * norm_column);

    // Get u norm squared
    norm_u_squared = u_0 * u_0 + mu2;

    // Get 2 * (1/u^T*u)
    beta1 = (2.0 / norm_u_squared);
    // endregion

#ifdef REPORT
    // Report x_0, sign(x_0), u[index_column], ||u[index_column,..,m]||_2^2, ||x[index_column + 1,..,m]||_2, u[index_column,..,m]
    file << "2. ----------- report x_0, sign(x_0), u[index_column], ||u[index_column,..,m]||_2^2, ||x[index_column + 1,..,m]||_2, u[index_column,..,m]\n";
    std::cout << "2. ----------- report x_0, sign(x_0), u[index_column], ||u[index_column,..,m]||_2^2, ||x[index_column + 1,..,m]||_2, u[index_column,..,m]\n";
    file << std::fixed << "x_0: " << x_0 << "\n";
    std::cout << std::fixed << "x_0: " << x_0 << "\n";
    file << std::fixed << "sign(x_0): " << sign_x_0 << "\n";
    std::cout << std::fixed << "sign(x_0): " << sign_x_0 << "\n";
    file << std::fixed << "u[index_column]: " << u[index_column] << "\n";
    std::cout << std::fixed << "u[index_column]: " << u[index_column] << "\n";
    file << std::fixed << "||u[index_column,..,m]||_2^2: " << norm_u_squared << "\n";
    std::cout << std::fixed << "||u[index_column,..,m]||_2^2: " << norm_u_squared << "\n";
    file << std::fixed << "||x[index_column,..,m]||_2: " << norm_column << "\n";
    std::cout << std::fixed << "||x[index_column,..,m]||_2: " << norm_column << "\n";
    file << std::fixed << std::setprecision(3) << "u[index_column,..,m]: \n";
    std::cout << std::fixed << std::setprecision(3) << "u[index_column,..,m]: \n";
    for (size_t index_of_u = index_column; index_of_u < A_height; ++index_of_u) {
      file << u[index_of_u] << " ";
      std::cout << u[index_of_u] << " ";
    }
    file << '\n';
    std::cout << '\n';
#endif

// ---------------------------------------------------------------------------------------------------------------------
// 7. Compute u^T*[R|B]
// 8. Store in v
// ---------------------------------------------------------------------------------------------------------------------
#pragma omp parallel for
    for (size_t index_columns_submatrix_A = index_column; index_columns_submatrix_A < A_width + A_height;
         ++index_columns_submatrix_A) {
      double tmp = 0.0;
      if(index_columns_submatrix_A < A_width){
        for (size_t index_rows_submatrix_A = index_column; index_rows_submatrix_A < A_height; ++index_rows_submatrix_A) {
          tmp += u[index_rows_submatrix_A] * R.elements[index_rows_submatrix_A * A_width + index_columns_submatrix_A];
        }
        v[index_columns_submatrix_A] = tmp;
      } else {
        for (size_t index_rows_submatrix_A = index_column; index_rows_submatrix_A < A_height; ++index_rows_submatrix_A) {
          tmp += u[index_rows_submatrix_A] * B.elements[index_rows_submatrix_A * A_height + (index_columns_submatrix_A - A_width)];
        }
        v[index_columns_submatrix_A] = tmp;
      }
    }

#ifdef REPORT
    // Report v^T = u^T*R
    file << "3. ----------- report v^T = u^T*R\n";
    std::cout << "3. ----------- report v^T = u^T*R\n";
    file << std::fixed << std::setprecision(3) << "v[index_column,..,m]: \n";
    std::cout << std::fixed << std::setprecision(3) << "v[index_column,..,m]: \n";
    for (size_t index_of_u = index_column; index_of_u < A_height; ++index_of_u) {
      file << v[index_of_u] << " ";
      std::cout << v[index_of_u] << " ";
    }
    file << '\n';
    std::cout << '\n';
#endif

// ---------------------------------------------------------------------------------------------------------------------
// 9. Compute R - 2 * (1/u^T*u) * u*v^T
// 8. Store in R
// ---------------------------------------------------------------------------------------------------------------------
#pragma omp parallel for
    for (size_t index_row_R = index_column; index_row_R < A_height; ++index_row_R) {
      for (size_t index_column_R = index_column; index_column_R < A_width + A_height; ++index_column_R) {
        if(index_column_R < A_width){
          R.elements[index_row_R * A_width + index_column_R] -= beta1 * u[index_row_R] * v[index_column_R];
        } else {
          B.elements[index_row_R * A_height + (index_column_R - A_width)] -= beta1 * u[index_row_R] * v[index_column_R];
        }
      }
    }

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

  // Delete dynamically allocated memory
  delete[] u;
  delete[] v;
}

/**
 * @brief QR decomposition using householder triangulation. R must be initialized before using this function
 *
 * @param A Matrix to be decomposed of m x n dimension
 * @param R Matrix upper triangular of m x n dimension
 */
void Thesis::QRDecompositionParallel(const Matrix &A, Matrix &R, std::ofstream &file) {
#ifdef DEBUG
  if (A.height >= A.width) {
    std::cout << "A podría tener columnas linealmente independientes\n";
  } else {
    std::cout << "A tiene columnas linealmente dependientes\n";
  }

  if (R.width != A.width || R.height != A.height) {
    throw std::runtime_error("R tiene dimensiones erroneas");
  }
#endif
  // -------------------------------------------------------------------------------------------------------------------
  // Copy A into R
  // region
  memcpy(R.elements, A.elements, A.width * A.height * sizeof(double));\
  // endregion

  // Constants of width and height
  const unsigned long A_width = A.width, A_height = A.height;

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

// ---------------------------------------------------------------------------------------------------------------------
// --------------------------------------------Householder triangulation------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------------

  // region Initialize number of iterations, u, v, holding variables and norm variables
  auto min = std::min(A.width, A.height);
  // u, v vectors and to be defined values
  auto *u = new double[A_height];
  auto *v = new double[A_height];
  double mu2, beta1, norm_column, norm_u_squared;
  // endregion

  for (size_t index_column = 0; index_column < min; ++index_column) {
#ifdef REPORT
    file
        << "-------------------------------- index_column " + std::to_string(index_column) + " --------------------------------\n";
    std::cout
        << "-------------------------------- index_column " + std::to_string(index_column) + " --------------------------------\n";
#endif
    // Not necesary to calculate every H with
    // v^T = u^T*R
    // A = R - 2 * (1/u^T*u) * u*v^T

    // Copy u
    norm_u_squared = 0.0;
    norm_column = 0.0;
    mu2 = 0.0;

// ---------------------------------------------------------------------------------------------------------------------
// 1. Copy elements of column vector of A at index_column to u omitting the first one
// 2. Compute norm squared of u omitting the first one
// ---------------------------------------------------------------------------------------------------------------------
    for (size_t index_of_u = index_column + 1; index_of_u < A_height; ++index_of_u) {
      u[index_of_u] = R.elements[index_of_u * A_width + index_column];
      double u_i = u[index_of_u];
      mu2 += u_i * u_i;
    }

#ifdef REPORT
    // Report u[index_column + 1,..,m]
    file << "1. ----------- report u[index_column + 1,..,m]\n";
    std::cout << "1. ----------- report u[index_column + 1,..,m]\n";
    file << std::fixed << "||u[index_column + 1,..,m]||_2^2: " << mu2 << "\n";
    std::cout << std::fixed << "||u[index_column + 1,..,m]||_2^2: " << mu2 << "\n";
    file << std::fixed << std::setprecision(3) << "u[index_column + 1,..,m]: \n";
    std::cout << std::fixed << std::setprecision(3) << "u[index_column + 1,..,m]: \n";
    for (size_t index_of_u = index_column + 1; index_of_u < A_height; ++index_of_u) {
      file << u[index_of_u] << " ";
      std::cout << u[index_of_u] << " ";
    }
    file << '\n';
    std::cout << '\n';
#endif
// ---------------------------------------------------------------------------------------------------------------------
// 3. Compute norm of the vector column of A at index_column
// 4. Compute the first element in u that inserts 0 where needed
// 5. Compute the squared norm of u
// 6. Compute the scalar \beta = 2 / ||u||_2^2
// ---------------------------------------------------------------------------------------------------------------------

    // region
    // Get a_i column norm
    double x_0 = R.elements[index_column * A_width + index_column];
    norm_column = x_0 * x_0 + mu2;
    norm_column = sqrt(norm_column);

    // Get u
    double sign_x_0 = x_0 >= 0.0 ? 1.0 : -1.0;
    double u_0 = u[index_column] = x_0 + (sign_x_0 * norm_column);

    // Get u norm squared
    norm_u_squared = u_0 * u_0 + mu2;

    // Get 2 * (1/u^T*u)
    beta1 = (2.0 / norm_u_squared);
    // endregion

#ifdef REPORT
    // Report x_0, sign(x_0), u[index_column], ||u[index_column,..,m]||_2^2, ||x[index_column + 1,..,m]||_2, u[index_column,..,m]
    file << "2. ----------- report x_0, sign(x_0), u[index_column], ||u[index_column,..,m]||_2^2, ||x[index_column + 1,..,m]||_2, u[index_column,..,m]\n";
    std::cout << "2. ----------- report x_0, sign(x_0), u[index_column], ||u[index_column,..,m]||_2^2, ||x[index_column + 1,..,m]||_2, u[index_column,..,m]\n";
    file << std::fixed << "x_0: " << x_0 << "\n";
    std::cout << std::fixed << "x_0: " << x_0 << "\n";
    file << std::fixed << "sign(x_0): " << sign_x_0 << "\n";
    std::cout << std::fixed << "sign(x_0): " << sign_x_0 << "\n";
    file << std::fixed << "u[index_column]: " << u[index_column] << "\n";
    std::cout << std::fixed << "u[index_column]: " << u[index_column] << "\n";
    file << std::fixed << "||u[index_column,..,m]||_2^2: " << norm_u_squared << "\n";
    std::cout << std::fixed << "||u[index_column,..,m]||_2^2: " << norm_u_squared << "\n";
    file << std::fixed << "||x[index_column,..,m]||_2: " << norm_column << "\n";
    std::cout << std::fixed << "||x[index_column,..,m]||_2: " << norm_column << "\n";
    file << std::fixed << std::setprecision(3) << "u[index_column,..,m]: \n";
    std::cout << std::fixed << std::setprecision(3) << "u[index_column,..,m]: \n";
    for (size_t index_of_u = index_column; index_of_u < A_height; ++index_of_u) {
      file << u[index_of_u] << " ";
      std::cout << u[index_of_u] << " ";
    }
    file << '\n';
    std::cout << '\n';
#endif

// ---------------------------------------------------------------------------------------------------------------------
// 7. Compute u^T*[R]
// 8. Store in v
// ---------------------------------------------------------------------------------------------------------------------
#pragma omp parallel for
    for (size_t index_columns_submatrix_A = index_column; index_columns_submatrix_A < A_width;
         ++index_columns_submatrix_A) {
      double tmp = 0.0;
      for (size_t index_rows_submatrix_A = index_column; index_rows_submatrix_A < A_height; ++index_rows_submatrix_A) {
        tmp += u[index_rows_submatrix_A] * R.elements[index_rows_submatrix_A * A_width + index_columns_submatrix_A];
      }
      v[index_columns_submatrix_A] = tmp;
    }

#ifdef REPORT
    // Report v^T = u^T*R
    file << "3. ----------- report v^T = u^T*R\n";
    std::cout << "3. ----------- report v^T = u^T*R\n";
    file << std::fixed << std::setprecision(3) << "v[index_column,..,m]: \n";
    std::cout << std::fixed << std::setprecision(3) << "v[index_column,..,m]: \n";
    for (size_t index_of_u = index_column; index_of_u < A_height; ++index_of_u) {
      file << v[index_of_u] << " ";
      std::cout << v[index_of_u] << " ";
    }
    file << '\n';
    std::cout << '\n';
#endif

// ---------------------------------------------------------------------------------------------------------------------
// 9. Compute R - 2 * (1/u^T*u) * u*v^T
// 8. Store in R
// ---------------------------------------------------------------------------------------------------------------------
#pragma omp parallel for
    for (size_t index_row_R = index_column; index_row_R < A_height; ++index_row_R) {
      for (size_t index_column_R = index_column; index_column_R < A_width; ++index_column_R) {
        R.elements[index_row_R * A_width + index_column_R] -= beta1 * u[index_row_R] * v[index_column_R];
      }
    }

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

  // Delete dynamically allocated memory
  delete[] u;
  delete[] v;
}

/**
 * @brief
 *
 * @param index_column
 * @param d_R
 * @param R_height
 * @param R_width
 * @param d_v
 * @param d_v_size
 * @param d_w
 * @param d_w_size
 */
__global__ void Thesis::CUDAInitializeHouseholderVector(unsigned long index_column,
                                                CUDAMatrix d_R,
                                                unsigned long R_height,
                                                unsigned long R_width,
                                                double *d_v,
                                                unsigned long d_v_size,
                                                double *d_w,
                                                unsigned long d_w_size) {
  unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;

  // w[index]
  if (index < R_height - index_column) {
    double tmp = 0.0;
    for (size_t index_row = 0; index_row < R_height - index_column; ++index_row) {
      tmp += d_R.elements[(index_row + index_column) * R_width + (index + index_column)] * d_v[index_row];
    }
    d_w[index] = tmp;
  }
}


/**
 * @brief Device function that calculates QR decomposition
 *
 * @param index_column index of A(i,i) to apply Householder transformation
 * @param d_R R upper triangular matrix result
 * @param R_height R.height
 * @param R_width R.width
 * @param d_v v column vector of householder transformation in the i-th column
 * @param d_v_size size of v
 * @param d_w w column vector of first part of calculation
 * @param d_w_size size of w
 */
__global__ void Thesis::CUDAQRDecompositionOnIndexColumn(double beta1,
                                                 unsigned long index_column,
                                                 CUDAMatrix d_R,
                                                 unsigned long R_height,
                                                 unsigned long R_width,
                                                 double *d_v,
                                                 unsigned long d_v_size,
                                                 double *d_w,
                                                 unsigned long d_w_size) {
  // Global row and column of result matrix
  unsigned long rowMatrix = index_column + threadIdx.y + blockIdx.y * blockDim.y;
  unsigned long colMatrix = index_column + threadIdx.x + blockIdx.x * blockDim.x;
  // Block indexing of rows and columns
//  int subRow = threadIdx.y;
//  int subCol = threadIdx.x;

  if (rowMatrix < R_height && colMatrix < R_width) {
    d_R.elements[rowMatrix * R_width + colMatrix] = index_column;
//    d_R.elements[rowMatrix * R_width + colMatrix] -=
//        beta1 * d_v[rowMatrix - index_column] * d_w[colMatrix - index_column];
  }
}

/**
 * @brief Device function that shows householder pattern
 *
 * @param index_column index of A(i,i) to apply Householder transformation
 * @param d_R R upper triangular matrix result
 * @param R_height R.height
 * @param R_width R.width
 */
__global__ void Thesis::test_CUDAQRDecompositionOnIndexColumn(unsigned long index_column,
                                                      CUDAMatrix d_R,
                                                      unsigned long R_height,
                                                      unsigned long R_width) {
  // Global row and column of result matrix
  unsigned long rowMatrix = index_column + threadIdx.y + blockIdx.y * blockDim.y;
  unsigned long colMatrix = index_column + threadIdx.x + blockIdx.x * blockDim.x;
  // Block indexing of rows and columns
//  int subRow = threadIdx.y;
//  int subCol = threadIdx.x;

  if (rowMatrix < R_height && colMatrix < R_width) {
    d_R.elements[rowMatrix * R_width + colMatrix] = index_column;
  }
}

/**
 * @brief QR decomposition using householder triangulation. Q and R must be initialized before using this function
 *
 * @param A Matrix to be decomposed of m x n dimension
 * @param Q Matrix that holds Householder multiplicacion transformations H_1H_2H_3...=Q of m x m dimension
 * @param R Matrix upper triangular of m x n dimension
 */
void Thesis::QRDecompositionCUDA(const Matrix &A, Matrix &R, std::ofstream &file) {
  if (A.height >= A.width) {
    std::cout << "A podría tener columnas linealmente independientes\n";
  } else {
    std::cout << "A tiene columnas linealmente dependientes\n";
//        throw std::runtime_error("A.height < A.width\n");
  }

  if (R.width != A.width || R.height != A.height) {
    throw std::runtime_error("R tiene dimensiones erroneas");
  }

  // Copy A into R
  memcpy(R.elements, A.elements, A.width * A.height * sizeof(double));

  // Make v vector
  auto v = new double[R.height];
  unsigned long A_height = A.height;
  unsigned long A_width = A.width;

  // Make CUDA Matrices
  CUDAMatrix d_R;
  d_R.height = R.height;
  d_R.width = R.width;
  cudaMalloc(&d_R.elements, d_R.height * d_R.width * sizeof(double));

  // Copy R int d_R
  cudaMemcpy(d_R.elements, R.elements, d_R.height * d_R.width * sizeof(double), cudaMemcpyHostToDevice);

  // Make array in device
  double *d_v, *d_w, norm_column = 0.0, norm_u_squared = 0.0, beta1 = 0.0;
  cudaMalloc(&d_v, d_R.height * sizeof(double));
  cudaMalloc(&d_w, d_R.height * sizeof(double));

#ifdef DEBUG
  // Report Matrix R
//    file << std::fixed << std::setprecision(3) << "R: \n";
//    std::cout << std::fixed << std::setprecision(3) << "R: \n";
//    for (size_t indexRow = 0; indexRow < A_height; ++indexRow) {
//      for (size_t indexCol = 0; indexCol < A_width; ++indexCol) {
//        file << R.elements[indexRow * A_width + indexCol] << " ";
//        std::cout << R.elements[indexRow * A_width + indexCol] << " ";
//      }
//      file << '\n';
//      std::cout << '\n';
//    }
#endif

  // TODO: Implementaré después la extracción de Q

  // Householder triangulation

  // Do householder transformation for every column
  auto min = std::min(A.width, A.height);
  for (size_t index_column = 0; index_column < min; ++index_column) {
    // Calculate norm of column and v
    double mu2 = 0.0;

    // Copy without the first element and store the norm squared without it.
    for (size_t index_of_u = 0; index_of_u < A_height - index_column; ++index_of_u) {
      v[index_of_u] = R.elements[(index_of_u + index_column) * A_width + index_column];
      double u_i = v[index_of_u];
      mu2 += u_i * u_i;
    }

    // Get a_i column norm
    double x_0 = R.elements[index_column * A_width + index_column];
    norm_column = x_0 * x_0 + mu2;
    norm_column = sqrt(norm_column);

    // Get u
    double sign_x_0 = x_0 >= 0.0 ? 1.0 : -1.0;
    double u_0 = v[index_column] = x_0 + (sign_x_0 * norm_column);

    // Get u norm squared
    norm_u_squared = u_0 * u_0 + mu2;

    // Get 2 * (1/u^T*u)
    beta1 = (2.0 / norm_u_squared);

    cudaMemcpy(d_v, v, R.height * sizeof(double), cudaMemcpyHostToDevice);

    CUDAInitializeHouseholderVector<<<(int) ceil((float) (d_R.width - index_column) / NT), NT>>>(index_column,
                                                                                                 d_R,
                                                                                                 d_R.height,
                                                                                                 d_R.width,
                                                                                                 d_v,
                                                                                                 R.height
                                                                                                     - index_column,
                                                                                                 d_w,
                                                                                                 R.height
                                                                                                     - index_column);
    cudaDeviceSynchronize();
    dim3 dimBlock(NT, NT);
    dim3 dimGrid((d_R.width + dimBlock.x - index_column - 1) / dimBlock.x,
                 (d_R.width + dimBlock.y - index_column - 1) / dimBlock.y);
    CUDAQRDecompositionOnIndexColumn<<<dimGrid, dimBlock>>>(beta1,
                                                            index_column,
                                                            d_R,
                                                            d_R.height,
                                                            d_R.width,
                                                            d_v,
                                                            R.height - index_column,
                                                            d_w,
                                                            R.height - index_column);
    cudaDeviceSynchronize();
  }

  // Copy R int d_R
  cudaMemcpy(R.elements, d_R.elements, R.height * R.width * sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree(d_R.elements);
  cudaFree(d_v);
  cudaFree(d_w);
  delete[] v;
}