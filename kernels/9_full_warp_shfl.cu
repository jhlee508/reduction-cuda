__forceinline__ __device__ double warpShuffle(double val, unsigned int mask = 0xffffffff) {
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    val += __shfl_down_sync(mask, val, offset);
  }
  return val;
}

__global__ void full_warp_shfl_kernel(double *arr, int size, double *res) {
  __shared__ double s_arr[32];

  int lid = threadIdx.x;
  int gid = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

  int lane = threadIdx.x % warpSize;
  int warpID = threadIdx.x / warpSize;
  
  double val = 0.0f;
  unsigned mask = 0xFFFFFFFFU;

  val = (gid < size) ? arr[gid] : 0;
  val += (gid + blockDim.x) < size ? arr[gid + blockDim.x] : 0;
  __syncthreads(); 

  // 1st warp-shuffle reduction
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    val += __shfl_down_sync(mask, val, offset);
  }
  if (lane == 0) 
    s_arr[warpID] = val;
  __syncthreads(); // Put warp results in shared mem

  // Hereafter, only 1 warp exists
  if (warpID == 0) {
    // Reload from SMEM if warp exists
    val = (lid < blockDim.x / warpSize) ? s_arr[lane] : 0;

    // Last warp-shuffle reduction
    val = warpShuffle(val, mask);
    
    if (lid == 0) { res[blockIdx.x] = val; }
  }
}