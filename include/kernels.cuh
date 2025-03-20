#pragma once

__global__ void interleaved_kernel(double* arr, int size, double* res);

__global__ void interleaved_contiguous_kernel(double* arr, int size, double* res);

__global__ void sequential_kernel(double* arr, int size, double* res);

__global__ void sequential_load_add_kernel(double* arr, int size, double* res);

__global__ void sequential_warp_shfl_kernel(double* arr, int size, double* res);

__global__ void sequential_unroll_last_kernel(double* arr, int size, double* res);

__global__ void sequential_unroll_all_kernel(double* arr, int size, double* res);

__global__ void sequential_unroll_all_multi_load_kernel(double* arr, int size, double* res);