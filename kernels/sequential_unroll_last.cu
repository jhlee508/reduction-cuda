__inline__ __device__ void warpReduceSum(volatile double* s_arr, int lid) {
  s_arr[lid] += s_arr[lid + 32];
  s_arr[lid] += s_arr[lid + 16];
  s_arr[lid] += s_arr[lid + 8];
  s_arr[lid] += s_arr[lid + 4];
  s_arr[lid] += s_arr[lid + 2];
  s_arr[lid] += s_arr[lid + 1];
}

__global__ void sequential_unroll_last_kernel(double* arr, int size, double* res) {
  extern __shared__ double s_arr[];

  int lid = threadIdx.x;
  int gid = blockIdx.x * blockDim.x + threadIdx.x;

  /* 1. Load to SMEM */
  s_arr[lid] = (gid < size) ? arr[gid] : 0;
  __syncthreads(); 

  /* 2. Reduction in SMEM */
  for (int s = blockDim.x / 2; s > 32; s >>= 1) { 
    if (lid < s) { 
      s_arr[lid] += s_arr[lid + s];
    }
    __syncthreads();
  }

  if (lid < warpSize)
    warpReduceSum(s_arr, lid);

  /* 3. Store to GMEM */
  if (lid == 0) { res[blockIdx.x] = s_arr[0]; }
}