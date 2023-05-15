//
// Created by andre on 2/03/23.
//

#ifndef SVD_THESIS_LIB_MATRIX_CUH_
#define SVD_THESIS_LIB_MATRIX_CUH_

struct Matrix{
  unsigned long width;
  unsigned long height;
  double *elements;

  Matrix(unsigned long height, unsigned long width){
    this->width = width;
    this->height = height;

    this->elements = new double [height * width];
  }

  ~Matrix(){
    delete []elements;
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
