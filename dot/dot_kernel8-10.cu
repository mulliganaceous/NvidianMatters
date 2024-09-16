/* ------------
 * This code is provided solely for the personal and private use of
 * students taking the CSC367H5 course at the University of Toronto.
 * Copying for purposes other than this use is expressly prohibited.
 * All forms of distribution of this code, whether as given or with
 * any changes, are expressly prohibited.
 *
 * Authors: Bogdan Simion, Felipe de Azevedo Piovezan
 *
 * All of the files in this directory and all subdirectories are:
 * Copyright (c) 2019 Bogdan Simion
 * -------------
 */

__inline__ __device__ float warpReduceSum(float val) { 
    for (int offset = warpSize/2; offset > 0; offset /= 2)
    {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__inline__ __device__ float blockReduceSum(float val) {
    static __shared__ int shared[32]; // Shared mem for 32 partial sums
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    val = warpReduceSum(val);     // Each warp performs partial reduction

    if (lane==0) shared[wid]=val; // Write reduced value to shared memory

    __syncthreads();              // Wait for all partial reductions

    //read from shared memory only if that warp existed
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

    if (wid==0) val = warpReduceSum(val); //Final reduce within first warp

    return val;
}

__global__ void dot_kernel8(float *g_idata1, float *g_idata2, float *g_odata,
                            int N) 
{
    float sum = 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
             i += blockDim.x * gridDim.x)
    {
        sum += g_idata1[i]*g_idata2[i];
    }

    sum = blockReduceSum(sum);
    if (threadIdx.x==0)
    {
        g_odata[blockIdx.x]=sum;
    }
}

__global__ void dot_kernel9(float *g_idata1, float *g_idata2, float *g_odata,
                            int N) 
{
    float sum = 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
             i += blockDim.x * gridDim.x)
    {
        sum += g_idata1[i]*g_idata2[i];
    }

    sum = warpReduceSum(sum);
    if ((threadIdx.x & (warpSize - 1)) == 0)
        atomicAdd(g_odata, sum);
}

__global__ void dot_kernel10(float *g_idata1, float *g_idata2, float *g_odata,
                             int N)
{
    float sum = 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
             i += blockDim.x * gridDim.x)
    {
        sum += g_idata1[i]*g_idata2[i];
    }
    sum = blockReduceSum(sum);
    if (threadIdx.x == 0)
        atomicAdd(g_odata, sum);
}
