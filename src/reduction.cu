#include <cstdio>

#include "reduction.h"

#define CHECK_CUDA(call)                                                 \
  do {                                                                   \
    cudaError_t status_ = call;                                          \
    if (status_ != cudaSuccess) {                                        \
      fprintf(stderr, "CUDA error (%s:%d): %s:%s\n", __FILE__, __LINE__, \
              cudaGetErrorName(status_), cudaGetErrorString(status_));   \
      exit(EXIT_FAILURE);                                                \
    }                                                                    \
  } while (0)


static double *output_cpu;
static double *output_gpu;

double reduction_naive(double* arr, int size) {
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
  sum = reduction_naive(arr, size);

  // (TODO) Launch kernel on a GPU

  // (TODO) Download sum from GPU

  // DO NOT REMOVE; NEED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());

  return sum;
}

void reduction_initialize(int num_intervals) {
  // (TODO) Allocate device memory

}

void reduction_finalize() {
  // (TODO) Free device memory

}