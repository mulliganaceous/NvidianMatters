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

#include <stdio.h>
#include <string>

/* This is your own kernel, so you should decide which parameters to 
   add here*/
void run_kernel5(const int8_t *filter, int32_t dimension, 
                 const int32_t *input, int32_t *output, 
                 int32_t width, int32_t height, int32_t smallest, int32_t largest) {
  // Figure out how to split the work into threads and call the kernel below.
  // There are dimension²*width*height such threads running.
  const unsigned THREADS_PER_BLOCK = 512;
  int threads = THREADS_PER_BLOCK;
  dim3 blocks((dimension*dimension*width*height + threads - 1)/threads);

  // launch kernel
  kernel5<<<blocks, threads, threads*sizeof(int32_t)>>>(filter, dimension, input, output, width, height);
  normalize5<<<(width*height + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(output, width, height, smallest, largest);
}

__global__ void kernel5(const int8_t *filter, int32_t dimension,
                        const int32_t *input, int32_t *output, 
                        int32_t width, int32_t height) {
  // Implement shared memory
  extern __shared__ int32_t sharedpix[]; 

  // Separate index into filter and pixel part
  int idx = blockIdx.x*blockDim.x + threadIdx.x; // thread index
  int pixelindex =  idx % (width*height);
  int filterindex = idx / (width*height);
  // Row and col indices
  int row = pixelindex / width; // < height
  int col = pixelindex % width; // < width
  int ip = filterindex / dimension;
  int jp = filterindex % dimension;
  int row_cur = row + ip - (dimension-1)/2;
  int col_cur = col + jp - (dimension-1)/2;
  if (pixelindex < width * height && row_cur >= 0 && row_cur < height && col_cur >= 0 && col_cur < width) {
    atomicAdd(&(output[pixelindex]), filter[dimension*ip + jp]*input[width*row_cur + col_cur]);
  }
  __syncthreads();
}

__global__ void normalize5(int32_t *image, int32_t width, int32_t height,
                           int32_t smallest, int32_t biggest) 
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (smallest == biggest || idx >= width*height)
    return;
  // Perform (no reduction needed)
  if (idx < width * height) {
    image[idx] = ((image[idx] - smallest)*255) / (biggest - smallest);
  }
  __syncthreads();
}
