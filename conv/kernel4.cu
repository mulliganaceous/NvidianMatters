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
#define THREADS 32

#include <stdio.h>
#include <string>

void run_kernel4(const int8_t *filter, int32_t dimension, 
                 const int32_t *input, int32_t *output,
                 int32_t width, int32_t height, int32_t smallest, int32_t largest) {
  // Figure out how to split the work into threads and call the kernel below.

  // launch kernel
  kernel4<<<1, THREADS>>>(filter, dimension, input, output, width, height);
  normalize4<<<1, THREADS>>>(output, width, height, smallest, largest);
}

__global__ void kernel4(const int8_t *filter, int32_t dimension,
                        const int32_t *input, int32_t *output, 
                        int32_t width, int32_t height) {
  int idx = blockIdx.x*blockDim.x + threadIdx.x; // offset
  int SIZE = width * height;

  // Perform (no reduction needed)
  if (idx < SIZE){
    // Each thread identified by idx does its own row up to height - 1
    for (int k = idx; k < SIZE; k += THREADS) {
      int pix = 0;
      int row = k/width; // < height
      int col = k%width; // < width
      // Perform filtering
      for (int ip = 0; ip < dimension; ip++) {
        for (int jp = 0; jp < dimension; jp++) {
          int row_cur = row + ip - (dimension-1)/2;
          int col_cur = col + jp - (dimension-1)/2;
          if (row_cur >= 0 && row_cur < height && col_cur >= 0 && col_cur < width) {
            pix += filter[dimension*ip + jp]*input[width*row_cur + col_cur];
          }
        }
      }
      output[width*row + col] = pix;
    }
    __syncthreads();
  }
}

__global__ void normalize4(int32_t *image, int32_t width, int32_t height,
                           int32_t smallest, int32_t biggest) 
{
  int idx = blockIdx.x*blockDim.x + threadIdx.x; // offset
  int SIZE = width * height;

  if (smallest == biggest || idx >= SIZE)
    return;
  for (int k = idx; k < SIZE; k += THREADS) {
    int row = k/width; // < height
    int col = k%width; // < width
    image[width*row + col] = ((image[width*row + col] - smallest) * 255) / (biggest - smallest);
  }
  __syncthreads();
}
