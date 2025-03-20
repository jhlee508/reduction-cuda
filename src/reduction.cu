#include <cstdio>
#include <cublas_v2.h>

#include "reduction.h"
#include "kernels.cuh"

static double* d_arr;
static double* d_output;

static cublasHandle_t handle;
static double* d_cublas_output;
static double* d_ones;

double cpu_single(double* arr, int size) {
  double sum = 0.0f;
  for (int i = 0; i < size; i++) { 
    sum += arr[i]; 
  }
  return sum;
}

double cpu_multithreading(double* arr, int size) {
  double sum = 0.0f;
#pragma omp parallel for reduction(+:sum)
  for (int i = 0; i < size; i++) { 
    sum += arr[i]; 
  }
  return sum;
}

double reduction_cpu(double* arr, int size) {
  double sum = 0.0;
  
  /* 1. CPU Single Core */
  // sum = cpu_single(arr, size);

  /* 2. CPU Multi-threading */
  // sum = cpu_multithreading(arr, size);

  return sum;
}

void reduction(double* arr, int size) {

  /* 3. Interleaved Addressing 
      - Level 0: CEIL_DIV(33554432, 256) = 131072 blocks
      - Level 1: CEIL_DIV(131072, 256) = 512 blocks
      - Level 2: CEIL_DIV(512, 256) = 2 blocks
      - Level 3: CEIL_DIV(2, 256) = 1 blocks
  */
  // dim3 gridDim0(CEIL_DIV(size, BLOCK_SIZE));
  // dim3 gridDim1(CEIL_DIV(gridDim0.x, BLOCK_SIZE));
  // dim3 gridDim2(CEIL_DIV(gridDim1.x, BLOCK_SIZE));
  // dim3 gridDim3(CEIL_DIV(gridDim2.x, BLOCK_SIZE));
  // interleaved_kernel<<<gridDim0, BLOCK_SIZE, BLOCK_SIZE * sizeof(double)>>>(d_arr, size, d_output);
  // interleaved_kernel<<<gridDim1, BLOCK_SIZE, BLOCK_SIZE * sizeof(double)>>>(d_output, gridDim0.x, d_output);
  // interleaved_kernel<<<gridDim2, BLOCK_SIZE, BLOCK_SIZE * sizeof(double)>>>(d_output, gridDim1.x, d_output);
  // interleaved_kernel<<<gridDim3, BLOCK_SIZE, BLOCK_SIZE * sizeof(double)>>>(d_output, gridDim2.x, d_output);

  /* 4. Interleaved Addressing (Contiguous) */
  // dim3 gridDim0(CEIL_DIV(size, BLOCK_SIZE));
  // dim3 gridDim1(CEIL_DIV(gridDim0.x, BLOCK_SIZE));
  // dim3 gridDim2(CEIL_DIV(gridDim1.x, BLOCK_SIZE));
  // dim3 gridDim3(CEIL_DIV(gridDim2.x, BLOCK_SIZE));
  // interleaved_contiguous_kernel<<<gridDim0, BLOCK_SIZE, BLOCK_SIZE * sizeof(double)>>>(d_arr, size, d_output);
  // interleaved_contiguous_kernel<<<gridDim1, BLOCK_SIZE, BLOCK_SIZE * sizeof(double)>>>(d_output, gridDim0.x, d_output);
  // interleaved_contiguous_kernel<<<gridDim2, BLOCK_SIZE, BLOCK_SIZE * sizeof(double)>>>(d_output, gridDim1.x, d_output);
  // interleaved_contiguous_kernel<<<gridDim3, BLOCK_SIZE, BLOCK_SIZE * sizeof(double)>>>(d_output, gridDim2.x, d_output);

  /* 5. Sequential Addressing 
      - Level 0: CEIL_DIV(33554432, 256) = 131072 blocks
      - Level 1: CEIL_DIV(131072, 256) = 512 blocks
      - Level 2: CEIL_DIV(512, 256) = 2 blocks
      - Level 3: CEIL_DIV(2, 256) = 1 blocks
  */
  // dim3 gridDim0(CEIL_DIV(size, BLOCK_SIZE));
  // dim3 gridDim1(CEIL_DIV(gridDim0.x, BLOCK_SIZE));
  // dim3 gridDim2(CEIL_DIV(gridDim1.x, BLOCK_SIZE));
  // dim3 gridDim3(CEIL_DIV(gridDim2.x, BLOCK_SIZE));
  // sequential_kernel<<<gridDim0, BLOCK_SIZE, BLOCK_SIZE * sizeof(double)>>>(d_arr, size, d_output);
  // sequential_kernel<<<gridDim1, BLOCK_SIZE, BLOCK_SIZE * sizeof(double)>>>(d_output, gridDim0.x, d_output);
  // sequential_kernel<<<gridDim2, BLOCK_SIZE, BLOCK_SIZE * sizeof(double)>>>(d_output, gridDim1.x, d_output);
  // sequential_kernel<<<gridDim3, BLOCK_SIZE, BLOCK_SIZE * sizeof(double)>>>(d_output, gridDim2.x, d_output);

  /* 6. Sequential Addressing (Warp Shuffle) 
      - Level 0: CEIL_DIV(33554432, 256) = 131072 blocks
      - Level 1: CEIL_DIV(131072, 256) = 512 blocks
      - Level 2: CEIL_DIV(512, 256) = 2 blocks
      - Level 3: CEIL_DIV(2, 256) = 1 blocks
  */
  // dim3 gridDim0(CEIL_DIV(size, BLOCK_SIZE));
  // dim3 gridDim1(CEIL_DIV(gridDim0.x, BLOCK_SIZE));
  // dim3 gridDim2(CEIL_DIV(gridDim1.x, BLOCK_SIZE));
  // dim3 gridDim3(CEIL_DIV(gridDim2.x, BLOCK_SIZE));
  // sequential_warp_shfl_kernel<<<gridDim0, BLOCK_SIZE, BLOCK_SIZE * sizeof(double)>>>(d_arr, size, d_output);
  // sequential_warp_shfl_kernel<<<gridDim1, BLOCK_SIZE, BLOCK_SIZE * sizeof(double)>>>(d_output, gridDim0.x, d_output);
  // sequential_warp_shfl_kernel<<<gridDim2, BLOCK_SIZE, BLOCK_SIZE * sizeof(double)>>>(d_output, gridDim1.x, d_output);
  // sequential_warp_shfl_kernel<<<gridDim3, BLOCK_SIZE, BLOCK_SIZE * sizeof(double)>>>(d_output, gridDim2.x, d_output);
  
  /* 7. Sequential Addressing (Unroll the Last Warp)
      - Level 0: CEIL_DIV(33554432, 256) = 131072 blocks
      - Level 1: CEIL_DIV(131072, 256) = 512 blocks
      - Level 2: CEIL_DIV(512, 256) = 2 blocks
      - Level 3: CEIL_DIV(2, 256) = 1 blocks
  */
  // dim3 gridDim0(CEIL_DIV(size, BLOCK_SIZE));
  // dim3 gridDim1(CEIL_DIV(gridDim0.x, BLOCK_SIZE));
  // dim3 gridDim2(CEIL_DIV(gridDim1.x, BLOCK_SIZE));
  // dim3 gridDim3(CEIL_DIV(gridDim2.x, BLOCK_SIZE));
  // sequential_unroll_last_kernel<<<gridDim0, BLOCK_SIZE, BLOCK_SIZE * sizeof(double)>>>(d_arr, size, d_output);
  // sequential_unroll_last_kernel<<<gridDim1, BLOCK_SIZE, BLOCK_SIZE * sizeof(double)>>>(d_output, gridDim0.x, d_output);
  // sequential_unroll_last_kernel<<<gridDim2, BLOCK_SIZE, BLOCK_SIZE * sizeof(double)>>>(d_output, gridDim1.x, d_output);
  // sequential_unroll_last_kernel<<<gridDim3, BLOCK_SIZE, BLOCK_SIZE * sizeof(double)>>>(d_output, gridDim2.x, d_output);

  /* 8. Sequential Addressing (Unroll All)
      - Level 0: CEIL_DIV(33554432, 256) = 131072 blocks
      - Level 1: CEIL_DIV(131072, 256) = 512 blocks
      - Level 2: CEIL_DIV(512, 256) = 2 blocks
      - Level 3: CEIL_DIV(2, 256) = 1 blocks
  */
  // dim3 gridDim0(CEIL_DIV(size, BLOCK_SIZE));
  // dim3 gridDim1(CEIL_DIV(gridDim0.x, BLOCK_SIZE));
  // dim3 gridDim2(CEIL_DIV(gridDim1.x, BLOCK_SIZE));
  // dim3 gridDim3(CEIL_DIV(gridDim2.x, BLOCK_SIZE));
  // sequential_unroll_all_kernel<<<gridDim0, BLOCK_SIZE, BLOCK_SIZE * sizeof(double)>>>(d_arr, size, d_output);
  // sequential_unroll_all_kernel<<<gridDim1, BLOCK_SIZE, BLOCK_SIZE * sizeof(double)>>>(d_output, gridDim0.x, d_output);
  // sequential_unroll_all_kernel<<<gridDim2, BLOCK_SIZE, BLOCK_SIZE * sizeof(double)>>>(d_output, gridDim1.x, d_output);
  // sequential_unroll_all_kernel<<<gridDim3, BLOCK_SIZE, BLOCK_SIZE * sizeof(double)>>>(d_output, gridDim2.x, d_output);

  /* 9. Sequential Addressing (Unroll All + Multiple Load) */
  dim3 gridDim0(CEIL_DIV(size, BLOCK_SIZE*2));
  dim3 gridDim1(CEIL_DIV(gridDim0.x, BLOCK_SIZE*2));
  dim3 gridDim2(CEIL_DIV(gridDim1.x, BLOCK_SIZE*2));
  dim3 gridDim3(CEIL_DIV(gridDim2.x, BLOCK_SIZE*2));
  sequential_unroll_all_multi_load_kernel<<<gridDim0, BLOCK_SIZE, BLOCK_SIZE * sizeof(double)>>>(d_arr, size, d_output);
  sequential_unroll_all_multi_load_kernel<<<gridDim1, BLOCK_SIZE, BLOCK_SIZE * sizeof(double)>>>(d_output, gridDim0.x, d_output);
  sequential_unroll_all_multi_load_kernel<<<gridDim2, BLOCK_SIZE, BLOCK_SIZE * sizeof(double)>>>(d_output, gridDim1.x, d_output);
  sequential_unroll_all_multi_load_kernel<<<gridDim3, BLOCK_SIZE, BLOCK_SIZE * sizeof(double)>>>(d_output, gridDim2.x, d_output);

  // DO NOT REMOVE; NEED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
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
    d_cublas_output, 1));      // Output: [1 x 1]
  
  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}

void reduction_cublas_v2(double* arr, int size) {
  CHECK_CUBLAS(cublasDdot(handle, size, d_arr, 1, d_ones, 1, d_cublas_output));

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}

void reduction_initialize(double* arr, int size) {
  CHECK_CUDA(cudaMalloc(&d_arr, size * sizeof(double)));
  CHECK_CUDA(cudaMemcpy(d_arr, arr, size * sizeof(double), cudaMemcpyHostToDevice));

  CHECK_CUDA(cudaMalloc(&d_output, size * sizeof(double)));
}

void cublas_initialize(int size) {
  CHECK_CUBLAS(cublasCreate(&handle));
  // CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH)); /* To enable TC */
  
  CHECK_CUDA(cudaMalloc(&d_cublas_output, sizeof(double)));

  CHECK_CUDA(cudaMalloc(&d_ones, size * sizeof(double)));
  double* h_ones = new double[size];
  for (int i = 0; i < size; i++) h_ones[i] = 1.0;
  cudaMemcpy(d_ones, h_ones, size * sizeof(double), cudaMemcpyHostToDevice);
  delete[] h_ones;
}

void reduction_finalize(double* output) {
  if (output != nullptr)
    CHECK_CUDA(cudaMemcpy(output, d_output, sizeof(double), cudaMemcpyDeviceToHost));

  CHECK_CUDA(cudaFree(d_arr));
  CHECK_CUDA(cudaFree(d_output));
}

void cublas_finalize(double* output) {
  if (output != nullptr)
    CHECK_CUDA(cudaMemcpy(output, d_cublas_output, sizeof(double), cudaMemcpyDeviceToHost));
  
  CHECK_CUDA(cudaFree(d_cublas_output));
  CHECK_CUDA(cudaFree(d_ones));

  CHECK_CUBLAS(cublasDestroy(handle));
}