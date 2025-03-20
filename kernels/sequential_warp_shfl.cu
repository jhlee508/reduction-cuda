__inline__ __device__ double warpShuffle(double val, unsigned int mask = 0xffffffff) {
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    val += __shfl_down_sync(mask, val, offset);
  }
  return val;
}

__global__ void sequential_warp_shfl_kernel(double* arr, int size, double* res) {
  extern __shared__ double s_arr[]; 

  int lid = threadIdx.x;                       
  int gid = blockIdx.x * blockDim.x + threadIdx.x; 

  /* 1. Load to SMEM */
  s_arr[lid] = (gid < size) ? arr[gid] : 0;
  __syncthreads();

  /* 2. Reduction in SMEM until blockDim.x > warpSize:
        (e.g., if blockDim.x == 128, then 128 -> 64 -> 32)
  */  
  for (int s = blockDim.x / 2; s >= warpSize; s >>= 1) {
    if (lid < s) {
      s_arr[lid] += s_arr[lid + s];
    }
    __syncthreads();
  }

  /* 3. Warp-level shuffle reduce for the final <= 32 threads */
  double sum = 0.0;
  if (lid < warpSize) {
    sum = warpShuffle(s_arr[lid]);
  }

  /* 4. Store to GMEM */
  if (lid == 0) { res[blockIdx.x] = sum; }
}