__global__ void sequential_kernel(double* arr, int size, double* res) {
  extern __shared__ double s_arr[];

  int lid = threadIdx.x;
  int gid = blockIdx.x * blockDim.x + threadIdx.x;

  /* 1. Load to SMEM */
  s_arr[lid] = (gid < size) ? arr[gid] : 0;
  __syncthreads(); 

  /* 2. Reduction in SMEM */
  for (int s = blockDim.x / 2; s > 0; s >>= 1) { // s = BLOCK_SIZE/2, BLOCK_SIZE/4, ..., 1
    if (lid < s) { // No bank conflict
      s_arr[lid] += s_arr[lid + s];
    }
    __syncthreads();
  }

  /* 3. Store to GMEM */
  if (lid == 0) { res[blockIdx.x] = s_arr[0]; }
}