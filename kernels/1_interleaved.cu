__global__ void interleaved_kernel(double* arr, int size, double* res) {
  extern __shared__ double s_arr[];

  int lid = threadIdx.x;
  int gid = blockIdx.x * blockDim.x + threadIdx.x;

  /* 1. Load to SMEM */
  s_arr[lid] = (gid < size) ? arr[gid] : 0;
  __syncthreads(); 

  /* 2. Reduction in SMEM */
  for (int s = 1; s < blockDim.x; s *= 2) { // s = 1, 2, 4, ..., (BLOCK_SIZE/2)
    if (lid % (2 * s) == 0) { // lid: (0, 2, ..., BLOCK_SIZE-2), (0, 4, ..., BLOCK_SIZE-4), ...
      s_arr[lid] += s_arr[lid + s];
    }
    __syncthreads();
  }

  /* 3. Store to GMEM */
  if (lid == 0) { res[blockIdx.x] = s_arr[0]; }
}