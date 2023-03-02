//
// Created by andre on 2/03/23.
//

#ifndef SVD_THESIS_LIB_MATRIX_CUH_
#define SVD_THESIS_LIB_MATRIX_CUH_

struct Matrix{
  unsigned long width;
  unsigned long height;
  double *elements;
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
};

struct CUDAVector{
  unsigned long length;
  double *elements;
};

#endif //SVD_THESIS_LIB_MATRIX_CUH_
