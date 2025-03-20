#pragma once

__global__ void interleaved_kernel(double* arr, int size, double* res);

__global__ void interleaved_kernel_v2(double* arr, int size, double* res);

__global__ void sequential_kernel(double* arr, int size, double* res);

__global__ void sequential_load_add_kernel(double* arr, int size, double* res);

__global__ void sequential_warp_shfl_kernel(double* arr, int size, double* res);