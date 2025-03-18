#include <getopt.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "reduction.h"
#include "util.h"

static void print_help(const char *prog_name) {
  printf("Usage: %s [-h] [-w] [-v] [-n 'num_iterations'] arr_size\n", prog_name);
  printf("Options:\n");
  printf("  -h: print this page.\n");
  printf("  -n: array size. (default: 100000).\n");
  printf("  -v: validate reduction. (default: off)\n");
  printf("  -w: warmup. (default: off)\n");
}

static int arr_size = 100000;
static int num_iterations = 1;
static bool warmup = false;
static bool validation = false;

static void parse_opt(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "n:wvh")) != -1) {
    switch (c) {
      case 'n': num_iterations = atoi(optarg); break;
      case 'w': warmup = true; break;
      case 'v': validation = true; break;
      case 'h':
      default: print_help(argv[0]); exit(0);
    }
  }
  for (int i = optind, j = 0; i < argc; ++i, ++j) {
    switch (j) {
      case 0: arr_size = atoi(argv[i]); break;
      default: break;
    }
  }
  printf("Options:\n");
  printf("  Array size: %d\n", arr_size);
  printf("  Number of iterations: %d\n", num_iterations);
  printf("  Warmup: %s\n", warmup ? "ON" : "OFF");
  printf("  Validation: %s\n", validation ? "ON" : "OFF");
  printf("\n");
}

int main(int argc, char **argv) {
  parse_opt(argc, argv);

  /* Allocate an array */
  printf("Allocating an array..."); fflush(stdout);
  double *arr = alloc_arr(arr_size);
  printf("Done!\n"); fflush(stdout);

  /* Initialize an array */
  printf("Initializing an array..."); fflush(stdout);
  rand_arr(arr, arr_size);
  printf("Done!\n"); fflush(stdout);

  /* Run cuBLAS reduction */
  /* Initialize cuBLAS */
  printf("Initializing cuBLAS..."); fflush(stdout);
  reduction_initialize(arr, arr_size);
  cublas_initialize(arr_size);
  printf("Done!\n"); fflush(stdout);
  /* Warmup cuBLAS */
  if (warmup) {
    printf("Warmup (cuBLAS)..."); fflush(stdout);
    reduction_cublas(arr, arr_size);
    reduction_cublas_v2(arr, arr_size);
    printf("Done!\n"); fflush(stdout);
  }
  /* Calculate cuBLAS performance */
  double cublas_elapsed_time_sum = 0.0;
  double cublas_output = 0.0;
  for (int i = 0; i < num_iterations; ++i) {
    printf("Calculating cuBLAS (iter=%d)...", i); fflush(stdout);
    double start_time = get_time();
    // reduction_cublas(arr, arr_size);
    reduction_cublas_v2(arr, arr_size);
    double elapsed_time = get_time() - start_time;
    printf("%f sec\n", elapsed_time);
    cublas_elapsed_time_sum += elapsed_time;
  }
  /* Finalize cuBLAS */
  printf("Finalizing cuBLAS..."); fflush(stdout);
  cublas_finalize(&cublas_output);
  reduction_finalize();
  printf("Done!\n"); fflush(stdout);

  /* Run my reduction */
  /* Initialize */
  printf("Initializing..."); fflush(stdout);
  reduction_initialize(arr, arr_size);
  printf("Done!\n"); fflush(stdout);
  /* Warmup */
  if (warmup) {
    printf("Warmup..."); fflush(stdout);
    reduction(arr, arr_size);
    printf("Done!\n"); fflush(stdout);
  }
  /* Calculate performance */
  double elapsed_time_sum = 0.0;
  double output = 0.0;
  for (int i = 0; i < num_iterations; ++i) {
    printf("Calculating (iter=%d)...", i); fflush(stdout);
    double start_time = get_time();
    output = reduction(arr, arr_size);
    double elapsed_time = get_time() - start_time;
    printf("%f sec\n", elapsed_time);
    elapsed_time_sum += elapsed_time;
  }   
  /* Finalize */
  printf("Finalizing..."); fflush(stdout);
  reduction_finalize();
  printf("Done!\n"); fflush(stdout);

  /* Validation */
  if (validation) {
    printf("Validating..."); fflush(stdout);
    check_reduction(arr, arr_size, output);
  }

  /* Print results */
  double cublas_elapsed_time_avg = cublas_elapsed_time_sum / num_iterations;
  printf("> Reduced Sum (cuBLAS): %.12f\n", cublas_output);
  printf("> Avg. Elapsed time (cuBLAS): %f sec\n", cublas_elapsed_time_avg);
  printf("> Avg. Bandwidth (cuBLAS): %.1f GB/s\n", 
    (double)arr_size * sizeof(double) / 1000000000 / cublas_elapsed_time_avg);
  
  double elapsed_time_avg = elapsed_time_sum / num_iterations;
  printf("> Reduced Sum: %.12f\n", output);
  printf("> Avg. Elapsed time: %f sec\n", elapsed_time_avg); 
  printf("> Avg. Bandwidth: %.1f GB/s\n", 
    (double)arr_size * sizeof(double) / 1000000000 / elapsed_time_avg);

  printf("> Perf. against cuBLAS: %.1f %%\n", 
    cublas_elapsed_time_avg / elapsed_time_avg * 100.0);

  return 0;
}