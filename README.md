# Optimizing Reduction using CUDA from Scratch
A step-by-step optimization of parallel sum reduction using CUDA to achieve GPU's peak memory bandwidth.


## Setup
### System
- 1 x AMD EPYC 7452 32-Core 2.35GHz CPU
- 1 x NVIDIA Tesla V100 32GB GPU (Peak GMEM bandwidth: `897` GB/s)
  - 4096-bit HBM2, 1752 MHz DDR = 4096 * 1752 / 8 = 897 (GB/s)

### Software
- CUDA Version: `12.4`


## Performance
The input array size is set to `33554432` (=`2^25`) doubles, and the number of threads in the thread block (`BLOCK_SIZE`) is set to `256`.

Implementation                                         | GB/s        | Memory BW Util. (%)
:------------------------------------------------------|:-----------:|:--------------------:
CPU (Single core)                                      | `7.9`       | 0.9
CPU (Multi-threading)                                  | `23.1`      | 2.6
1: Interleaved Addressing                              | `303.0`     | 33.8
cuBLAS (GEMV)                                          | `402.4`     | 44.9
2: Interleaved Addressing (+ Contiguous Thread)        | `427.0`     | 47.6
cuBLAS (DOT)                                           | `446.5`     | 49.8
3: Sequential Addressing                               | `649.5`     | 72.4
4: Sequential Addressing (+ Multiple Load per Thread)  | `872.6`     | 97.3
5: Sequential Addressing (+ Warp Shuffle Last Warp)    | `876.8`     | 97.7
6: Sequential Addressing (+ Unroll Last Warp)          | `879.8`     | 98.1
7: Sequential Addressing (+ Unroll All)                | `884.4`     | 98.6
8: Sequential Addressing (+ Final Tuning)              | `896.1`     | 99.9
0: Peak Memory Bandwidth                               | `897`       | 100

cf. When using cuBLAS to perform the sum reduction, it’s actually executing a GEMV or DOT operation that reads from two separate memory buffers (the input array and the “ones” vector). This effectively doubles the amount of data that must be transferred from memory and halves the achievable bandwidth compared to a true single-input reduction.


## Usage
### Build
```bash
$ make
```
### Run
```bash
$ ./main -w -n 5 -v 33554432
```
```
Usage: main [-h] [-w] [-v] [-n 'num_iterations'] <arr_size>
Options:
  -h: print this page.
  -w: warmup. (default: off)
  -v: validate reduction. (default: off)
  -n: number of iterations. (default: 1)
  <arr_size>: size of the array. (default: 100000)
```

## References
- https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
- https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
- https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html
- https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/