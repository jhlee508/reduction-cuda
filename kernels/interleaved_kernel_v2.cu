__global__ void interleaved_kernel_v2(double* arr, int size, double* res) {
  extern __shared__ double s_arr[];

  int lid = threadIdx.x;
  int gid = blockIdx.x * blockDim.x + threadIdx.x;

  /* 1. Load to SMEM */
  s_arr[lid] = (gid < size) ? arr[gid] : 0;
  __syncthreads(); 

  /* 2. Reduction in SMEM */
  for (int s = 1; s < blockDim.x; s *= 2) { // s = 1, 2, 4, ..., (BLOCK_SIZE/2)
    int idx = (2 * lid) * s; // Eliminate modulo operation w/ strided access
    if (idx < blockDim.x) { // No branch divergence
      s_arr[idx] += s_arr[idx + s];
    }
    __syncthreads();
  }

  /* 3. Store to GMEM */
  if (lid == 0) { res[blockIdx.x] = s_arr[0]; }
}