//
// Created by andre on 2/03/23.
//

#ifndef SVD_THESIS_LIB_HOUSEHOLDERMETHODS_CUH_
#define SVD_THESIS_LIB_HOUSEHOLDERMETHODS_CUH_

struct Matrix;
struct CUDAMatrix;

#include <fstream>
#include <iostream>
#include <iomanip>
#include "global.cuh"
#include "Matrix.cuh"

/**
 * @brief QR decomposition using householder triangulation. B and R must be initialized before using this function.
 *
 * @param A Matrix to be decomposed of m x n dimension
 * @param B Matrix that holds Householder multiplicacion transformations of the form H_1H_2H_3...B of m x m dimension
 * @param R Matrix upper triangular of m x n dimension
 */
void QRDecompositionParallelWithB(const Matrix &A, Matrix &B, Matrix &R, std::ofstream &file);

/**
 * @brief QR decomposition using householder triangulation. R must be initialized before using this function
 *
 * @param A Matrix to be decomposed of m x n dimension
 * @param R Matrix upper triangular of m x n dimension
 */
void QRDecompositionParallel(const Matrix &A, Matrix &R, std::ofstream &file);

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
__global__ void CUDAInitializeHouseholderVector(unsigned long index_column,
                                                CUDAMatrix d_R,
                                                unsigned long R_height,
                                                unsigned long R_width,
                                                double *d_v,
                                                unsigned long d_v_size,
                                                double *d_w,
                                                unsigned long d_w_size);


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
__global__ void CUDAQRDecompositionOnIndexColumn(double beta1,
                                                 unsigned long index_column,
                                                 CUDAMatrix d_R,
                                                 unsigned long R_height,
                                                 unsigned long R_width,
                                                 double *d_v,
                                                 unsigned long d_v_size,
                                                 double *d_w,
                                                 unsigned long d_w_size);

/**
 * @brief Device function that shows householder pattern
 *
 * @param index_column index of A(i,i) to apply Householder transformation
 * @param d_R R upper triangular matrix result
 * @param R_height R.height
 * @param R_width R.width
 */
__global__ void test_CUDAQRDecompositionOnIndexColumn(unsigned long index_column,
                                                      CUDAMatrix d_R,
                                                      unsigned long R_height,
                                                      unsigned long R_width);

/**
 * @brief QR decomposition using householder triangulation. Q and R must be initialized before using this function
 *
 * @param A Matrix to be decomposed of m x n dimension
 * @param Q Matrix that holds Householder multiplicacion transformations H_1H_2H_3...=Q of m x m dimension
 * @param R Matrix upper triangular of m x n dimension
 */
void QRDecompositionCUDA(const Matrix &A, Matrix &R, std::ofstream &file);

#endif //SVD_THESIS_LIB_HOUSEHOLDERMETHODS_CUH_
