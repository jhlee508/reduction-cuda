__global__ void sequential_load_add_kernel(double* arr, int size, double* res) {
	extern __shared__ double s_arr[];

	int lid = threadIdx.x;
	int gid = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

	/* 1. Load to SMEM */
	// s_arr[lid] = (gid < size) ? arr[gid] : 0;
	// Two loads and first add of the reduction
	s_arr[lid] = (gid < size) ? arr[gid] : 0;
	s_arr[lid] += (gid + blockDim.x) < size ? arr[gid + blockDim.x] : 0;
	__syncthreads();

	for (int s  = blockDim.x / 2; s > 0; s >>= 1) { // s = BLOCK_SIZE/4, BLOCK_SIZE/8, ..., 1
		if (lid < s) { // No bank conflict
			s_arr[lid] += s_arr[lid + s];
		}
		__syncthreads();
	}

	/* 3. Store to GMEM */
	if (lid == 0) { 
		res[blockIdx.x] = s_arr[0]; 
	}
}