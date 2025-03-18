#include <cstdio>
#include <cublas_v2.h>

#include "reduction.h"

static double *d_arr, *d_output;
static double *d_ones;
static cublasHandle_t handle;

double reduction_cpu(double* arr, int size) {
  double sum = 0.0f;
#pragma omp parallel for reduction(+:sum)
  for (int i = 0; i < size; i++) { 
    sum += arr[i]; 
  }
  return sum;
}

double reduction(double* arr, int size) {
  double sum = 0.0;
  // Remove this line after you complete the matmul on GPU
  sum = reduction_cpu(arr, size);

  // (TODO) Launch kernel on a GPU

  // (TODO) Download sum from GPU

  // DO NOT REMOVE; NEED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());

  return sum;
}

void reduction_cublas(double* arr, int size) {
  const double alpha = 1.0, beta = 0.0;

  /* y [1 x 1] = A^T [1 x size] * x [size x 1] */
  CHECK_CUBLAS(cublasDgemv(handle, 
    CUBLAS_OP_T,        // To transpose
    size, 1,            // Matrix: [size x 1]
    &alpha,
    d_arr, size,        // lda = size
    d_ones, 1,          // Vector: [size x 1]
    &beta,
    d_output, 1));      // Output: [1 x 1]
  
  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}

void reduction_cublas_v2(double* arr, int size) {
  CHECK_CUBLAS(cublasDdot(handle, size, d_arr, 1, d_ones, 1, d_output));

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}

void reduction_initialize(double* arr, int size) {
  CHECK_CUDA(cudaMalloc(&d_arr, size * sizeof(double)));
  CHECK_CUDA(cudaMemcpy(d_arr, arr, size * sizeof(double), cudaMemcpyHostToDevice));
}

void cublas_initialize(int size) {
  CHECK_CUBLAS(cublasCreate(&handle));
  // CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH)); /* To enable TC */
  
  CHECK_CUDA(cudaMalloc(&d_ones, size * sizeof(double)));
  double* h_ones = new double[size];
  for (int i = 0; i < size; i++) h_ones[i] = 1.0;
  cudaMemcpy(d_ones, h_ones, size * sizeof(double), cudaMemcpyHostToDevice);
  delete[] h_ones;
  
  CHECK_CUDA(cudaMalloc(&d_output, sizeof(double)));
}

void reduction_finalize() {
  CHECK_CUDA(cudaFree(d_arr));
}

void cublas_finalize(double* output) {
  if (output != nullptr)
    CHECK_CUDA(cudaMemcpy(output, d_output, sizeof(double), cudaMemcpyDeviceToHost));
  
  CHECK_CUDA(cudaFree(d_ones));
  CHECK_CUDA(cudaFree(d_output));

  CHECK_CUBLAS(cublasDestroy(handle));
}