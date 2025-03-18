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

  /* Warmup */
  printf("Warmup... "); fflush(stdout);
  for (int i = 0; i < 3; ++i) {
    reduction_initialize(arr_size);
    reduction(arr, arr_size);
    reduction_finalize();
  }
  printf("Done!\n"); fflush(stdout);

  /* Initialize an array */
  printf("Initializing an array..."); fflush(stdout);
  rand_arr(arr, arr_size);
  printf("Done!\n"); fflush(stdout);

  /* Initialize to run reduction */
  printf("Initializing..."); fflush(stdout);
  reduction_initialize(arr_size);
  printf("Done!\n"); fflush(stdout);

  /* Run actual computation */
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
  printf("> Reduced Sum: %.16f\n", output);
  printf("> Avg. Elapsed time: %.3f sec\n", elapsed_time_sum / num_iterations); 
  printf("> Avg. Bandwidth: %.3f GB/s\n", 
    (double)arr_size * sizeof(double) / (1 << 30) / (elapsed_time_sum / num_iterations));  

  return 0;
}