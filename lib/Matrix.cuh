//
// Created by andre on 2/03/23.
//

#ifndef SVD_THESIS_LIB_MATRIX_CUH_
#define SVD_THESIS_LIB_MATRIX_CUH_

#include <stdlib.h>

struct Matrix{
  unsigned long width;
  unsigned long height;
  double *elements;

  Matrix(unsigned long height, unsigned long width){
    this->width = width;
    this->height = height;

//    cudaMallocHost((void**) &this->elements, height * width * sizeof(double));
    this->elements = new double [height * width];
//    this->elements = (double *)malloc(height * width * sizeof(double));
  }

  void freeHost(){
//    cudaFreeHost(this->elements);
  }

  ~Matrix(){
//    cudaFreeHost(this->elements);
    delete []elements;
//    free(this->elements);
  }
};

struct Vector{
  unsigned long length;
  double *elements;
  ~Vector(){
    delete []elements;
  }
};

struct CUDAMatrix{
  unsigned long width;
  unsigned long height;
  double *elements;

  CUDAMatrix(unsigned long height, unsigned long width){
    this->width = width;
    this->height = height;

    cudaMalloc(&this->elements, height * width * sizeof(double));
    cudaMemset(&this->elements, 0, height * width * sizeof(double));
  }

  CUDAMatrix(double *arr, size_t length){
    this->height = 1;
    this->width = length;
    cudaMalloc(&this->elements, length * sizeof(double));
    cudaMemcpy(this->elements, arr, length * sizeof(double),
               cudaMemcpyHostToDevice);
  }

  explicit CUDAMatrix(Matrix &matrix){
    this->width = matrix.width;
    this->height = matrix.height;

    cudaMalloc(&this->elements, height * width * sizeof(double));
    cudaMemcpy(this->elements, matrix.elements, this->height * this->width * sizeof(double),
               cudaMemcpyHostToDevice);
  }

  void copy_to_host(Matrix &matrix) const{
    cudaMemcpy(matrix.elements,
               this->elements,
               matrix.width * matrix.height * sizeof(double),
               cudaMemcpyDeviceToHost);
  }

  void copy_to_host(double *arr, size_t length) const{
    cudaMemcpy(arr, this->elements, length * sizeof(double), cudaMemcpyDeviceToHost);
  }

  void copy_from_host(Matrix &matrix) const{
    cudaMemcpy(this->elements,
               matrix.elements,
               width * height * sizeof(double),
               cudaMemcpyHostToDevice);
  }

  void copy_from_device(CUDAMatrix &matrix) const{
    cudaMemcpy(this->elements,
               matrix.elements,
               width * height * sizeof(double),
               cudaMemcpyDeviceToDevice);
  }

  void free() const{
    cudaFree(this->elements);
  }
};

struct CUDAVector{
  unsigned long length;
  double *elements;
};

#endif //SVD_THESIS_LIB_MATRIX_CUH_
