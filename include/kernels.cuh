#pragma once

__global__ void interleaved_kernel(double* arr, int size, double* res);

__global__ void interleaved_contiguous_kernel(double* arr, int size, double* res);

__global__ void sequential_kernel(double* arr, int size, double* res);

__global__ void sequential_multi_load_kernel(double* arr, int size, double* res);

__global__ void sequential_warp_shfl_last_kernel(double* arr, int size, double* res);

__global__ void sequential_unroll_last_kernel(double* arr, int size, double* res);

__global__ void sequential_unroll_all_kernel(double* arr, int size, double* res);

__global__ void sequential_tuning_kernel(double* arr, int size, double* res);

__global__ void sequential_atomic_kernel(double* arr, int size, double* res);

__global__ void full_warp_shfl_kernel(double* arr, int size, double* res);