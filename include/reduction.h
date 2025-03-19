#pragma once

#include <cstdlib>

#define CHECK_CUDA(call)                                                 \
  do {                                                                   \
    cudaError_t status_ = call;                                          \
    if (status_ != cudaSuccess) {                                        \
      fprintf(stderr, "CUDA error (%s:%d): %s:%s\n", __FILE__, __LINE__, \
              cudaGetErrorName(status_), cudaGetErrorString(status_));   \
      exit(EXIT_FAILURE);                                                \
    }                                                                    \
  } while (0)

#define CHECK_CUBLAS(call)                                                   \
  do {                                                                       \
    cublasStatus_t status_ = call;                                           \
    if (status_ != CUBLAS_STATUS_SUCCESS) {                                  \
      fprintf(stderr, "CUBLAS error (%s:%d): %s, %s\n", __FILE__, __LINE__,  \
              cublasGetStatusName(status_), cublasGetStatusString(status_)); \
      exit(EXIT_FAILURE);                                                    \
    }                                                                        \
  } while (0)

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

#define BLOCK_SIZE 128
#define BLOCK_SIZE_2 256
#define BLOCK_SIZE_3 512
#define BLOCK_SIZE_4 1024

double reduction_cpu(double* arr, int size);

void reduction(double* arr, int size);

void reduction_cublas(double* arr, int size);

void reduction_cublas_v2(double* arr, int size);

void reduction_initialize(double* arr, int size);

void cublas_initialize(int size);

void reduction_finalize(double *output);

void cublas_finalize(double *output);