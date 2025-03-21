__inline__ __device__ void warpReduceSum(volatile double* s_arr, int lid) {
  s_arr[lid] += s_arr[lid + 32];
  s_arr[lid] += s_arr[lid + 16];
  s_arr[lid] += s_arr[lid + 8];
  s_arr[lid] += s_arr[lid + 4];
  s_arr[lid] += s_arr[lid + 2];
  s_arr[lid] += s_arr[lid + 1];
}

__global__ void sequential_tuning_kernel(double* arr, int size, double* res) {
  extern __shared__ double s_arr[];

  int lid = threadIdx.x;
  int gid = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

  /* 1. Load to SMEM */
  s_arr[lid] = 0;
  if (gid < size) { s_arr[lid] = arr[gid]; }
  if (gid + blockDim.x < size) { s_arr[lid] += arr[gid + blockDim.x]; }
  __syncthreads();

  /* 2. Reduction in SMEM (only when the BLOCK_SIZE is 256!) */
  if (lid < 128) { s_arr[lid] += s_arr[lid + 128]; } __syncthreads();
  if (lid < 64) { s_arr[lid] += s_arr[lid + 64]; } __syncthreads();
  
  if (lid < warpSize)
    warpReduceSum(s_arr, lid);

  /* 3. Store to GMEM */
  if (lid == 0) { res[blockIdx.x] = s_arr[0]; }
}