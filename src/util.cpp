#include "util.h"

#include <sys/time.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                 \
  do {                                                                   \
    cudaError_t status_ = call;                                          \
    if (status_ != cudaSuccess) {                                        \
      fprintf(stderr, "CUDA error (%s:%d): %s:%s\n", __FILE__, __LINE__, \
              cudaGetErrorName(status_), cudaGetErrorString(status_));   \
      exit(EXIT_FAILURE);                                                \
    }                                                                    \
  } while (0)

double get_time() {
  struct timespec tv;
  clock_gettime(CLOCK_MONOTONIC, &tv);
  return tv.tv_sec + tv.tv_nsec * 1e-9;
}

void check_reduction(double* arr, int arr_size, double output) {
  double ans = 0.0;

#pragma omp parallel for reduction(+:ans)
  for (int i = 0; i < arr_size; ++i) {
    ans += arr[i];
  }

  const double eps = 1e-6;
  bool is_valid = fabs(ans - output) < eps || 
                  (ans != 0 && fabs((ans - output) / ans) <= eps);

  if (is_valid) { 
    printf("PASSED!\n"); 
  } 
  else { 
    printf("FAILED!\n"); 
    printf("Correct value: %.12f, Your value: %.12f\n", ans, output);
  }
}

double* alloc_arr(int arr_size) {
  double *a;
  CHECK_CUDA(cudaMallocHost(&a, sizeof(double) * arr_size));
  return a;
}

void rand_arr(double* arr, int arr_size) {
  srand(123);
  for (int i = 0; i < arr_size; ++i) {
    arr[i] = (double)rand() / RAND_MAX - 0.5;
  }
}