__forceinline__ __device__ double warpShuffleSum(double sum, unsigned int mask = 0xffffffff) {
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    sum += __shfl_down_sync(mask, sum, offset);
  }
  return sum;
}

__global__ void full_warp_shfl_kernel(double *arr, int size, double *res) {
  __shared__ double s_arr[8]; // the number of warps in a block (256 / 32 = 8)

  int lid = threadIdx.x;
  int gid = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

  int laneId = threadIdx.x % warpSize;
  int warpId = threadIdx.x / warpSize;
  
  double sum = 0.0f;
  sum = (gid < size) ? arr[gid] : 0;
  sum += (gid + blockDim.x) < size ? arr[gid + blockDim.x] : 0;
  __syncthreads(); 

  /* First warp-shuffle reduction */
  sum = warpShuffleSum(sum, 0xffffffff);

  /* Each lane stores the partial sum to SMEM */
  if (laneId == 0) { s_arr[warpId] = sum; }
  __syncthreads(); 

  /* Last warp-shuffle reduction using only the first warp */
  if (warpId == 0) {
    // Read SMEM with the first warp (but only threads less than numWarpsPerBlock)
    int numWarpsPerBlock = blockDim.x / warpSize; // 256 / 32 = 8
    sum = (lid < numWarpsPerBlock) ? s_arr[laneId] : 0;

    sum = warpShuffleSum(sum, 0xffffffff);
  }

  if (lid == 0) { res[blockIdx.x] = sum; }
}