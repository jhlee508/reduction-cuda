#pragma once

#include <cstdlib>

#define THREADS_PER_BLOCK 1024
#define ELEMENTS_PER_BLOCK (THREADS_PER_BLOCK * 2)


double reduction_naive(double* arr, int arr_size);

void reduction_initialize(int num_intervals);

double reduction(double* arr, int arr_size);

void reduction_finalize();