# Optimizing Reduction using CUDA from Scratch
A step-by-step optimization of sum reduction using CUDA to achieve near GPU's peak memory bandwidth.


## Setup
### System
- 1 x AMD EPYC 7452 32-Core 2.35GHz CPU
- 1 x NVIDIA Tesla V100 32GB GPU (Peak GMEM bandwidth: `897` GB/s)
  - 4096-bit HBM2, 1752 MHz DDR = 4096 * 1752 / 8 = 897 (GB/s)

### Software
- CUDA Version: `12.4`

## Performance
The input array size is set to `33554432` (=`2^25`) doubles.

Implementation                           | GB/s        | Memory BW Util. (%)
---------------------------------------- | ----------- | --------------------
1: CPU (Single core)                     | `7.9`       | 0.9
2: CPU (Multi-threading)                 | `23.1`      | 2.6
3: Interleaved Addressing                | `340.6`     | 38.0
cuBLAS (GEMV)                            | `402.4`     | 44.9
4: Interleaved Addressing (Contiguous)   | `435.4`     | 48.5
cuBLAS (DOT)                             | `446.5`     | 49.8
5: Sequential Addressing                 | `649.5`     | 72.4
6: Sequential Addressing (Warp Shuffle)  | `720.1`     | 80.3
7:
0: Peak Memory Bandwidth                 | `897`       | 100

cf. When using cuBLAS to perform the sum reduction, it’s actually executing a GEMV or DOT operation that reads from two separate memory buffers (the input array and the “ones” vector). This effectively doubles the amount of data that must be transferred from memory and halves the achievable bandwidth compared to a true single-input reduction.

## Usage
### Build
```bash
$ make -j8
```
### Run
```bash
$ ./main -w -n 5 -v 33554432
```
```
Usage: main [-h] [-w] [-v] [-n 'num_iterations'] <arr_size>
Options:
  -h: print this page.
  -n: array size. (default: 100000).
  -v: validate reduction. (default: off)
  -w: warmup. (default: off)
```

## References
- https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
- https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
- https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html
- https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/