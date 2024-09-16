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

#include "kernels.h"

template <unsigned int blockSize>
__global__ void dot_kernel6(float *g_idata1, float *g_idata2, float *g_odata) {
    // Shared memory
    extern __shared__ int sdata[];
    
    // tid and global id counters
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x*2) + threadIdx.x;

    // Perform
    sdata[tid] = g_idata1[i]*g_idata2[i] + g_idata1[i+blockDim.x]*g_idata2[i+blockDim.x];
    __syncthreads();

    // do reduction in shared memory. A loop is ddone for cases where s > warp size.
	if (blockSize >= 512) { if (tid < 256){ sdata[tid] += sdata[tid + 256];} __syncthreads();}
	if (blockSize >= 256) { if (tid < 128){ sdata[tid] += sdata[tid + 128];} __syncthreads();}
	if (blockSize >= 128) {	if (tid <  64){ sdata[tid] += sdata[tid +  64];} __syncthreads();}
	if (tid < 32) {
		volatile int* smem = sdata;
		if (blockSize >= 64) smem[tid] += smem[tid + 32];
		if (blockSize >= 32) smem[tid] += smem[tid + 16];
		if (blockSize >= 16) smem[tid] += smem[tid + 8];
		if (blockSize >=  8) smem[tid] += smem[tid + 4];
		if (blockSize >=  4) smem[tid] += smem[tid + 2];
		if (blockSize >=  2) smem[tid] += smem[tid + 1];
	} 

    // Write result for this block back to global memory
    if (tid == 0) { g_odata[blockIdx.x] = sdata[0]; }
}
