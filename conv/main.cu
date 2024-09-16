/* -----------
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

#include <stdio.h>
#include <string>
#include <unistd.h>

#include "pgm.h"
#include "kernels.h"
#define DEBUGMODE 1
#define MATRIXPRINT(...) if (DEBUGMODE/2) printf(__VA_ARGS__)
#define ERRORPRINT(...) if (DEBUGMODE%2) fprintf(stderr, __VA_ARGS__)
// filter settings
const int8_t FILTER[] = {
    0,  1,  1,  2,  2,  2,  1,  1,  0, 
    1,  2,  4,  5,  5,  5,  4,  2,  1, 
    1,  4,  5,  3,  0,  3,  5,  4,  1, 
    2,  5,  3,-12,-24,-12,  3,  5,  2, 
    2,  5,  0,-24,-40,-24,  0,  5,  2, 
    2,  5,  3,-12,-24,-12,  3,  5,  2,
    1,  4,  5,  3,  0,  3,  5,  4,  1, 
    1,  2,  4,  5,  5,  5,  4,  2,  1, 
    0,  1,  1,  2,  2,  2,  1,  1,  0,
};
const int FILTER_DIMENSION = 9;

/* Use this function to print the time of each of your kernels.
 * The parameter names are intuitive, but don't hesitate to ask
 * for clarifications.
 * DO NOT modify this function.*/
void print_run(float time_cpu, int kernel, float time_gpu_computation,
               float time_gpu_transfer_in, float time_gpu_transfer_out) {
  printf("%12.6f ", time_cpu);
  printf("%5d ", kernel);
  printf("%12.6f ", time_gpu_computation);
  printf("%14.6f ", time_gpu_transfer_in);
  printf("%15.6f ", time_gpu_transfer_out);
  printf("%13.2f ", time_cpu / time_gpu_computation);
  printf("%7.2f\n", time_cpu / (time_gpu_computation + time_gpu_transfer_in +
                                time_gpu_transfer_out));
}

int main(int argc, char **argv) {
  int c;
  std::string input_filename, cpu_output_filename, base_gpu_output_filename;
  if (argc < 3) {
    printf("Wrong usage. Expected -i <input_file> -o <output_file>\n");
    return 0;
  }
  while ((c = getopt(argc, argv, "i:o:")) != -1) {
    switch (c) {
    case 'i':
      input_filename = std::string(optarg);
      break;
    case 'o':
      cpu_output_filename = std::string(optarg);
      base_gpu_output_filename = std::string(optarg);
      break;
    default:
      return 0;
    }
  }

  /* Input step */
  pgm_image source_img;
  init_pgm_image(&source_img);
  if (load_pgm_from_file(input_filename.c_str(), &source_img) != NO_ERR) {
    printf("Error loading source image.\n");
    return 0;
  }

  /* Do not modify this printf */
  printf("CPU_time(ms) Kernel GPU_time(ms) TransferIn(ms) TransferOut(ms) "
         "Speedup_noTrf Speedup\n");

  /* CPU implementation */
  cudaEvent_t start, stop;
  float computation_time, transfer_in, transfer_out, time_cpu;
  computation_time = 0;
  transfer_in = 0;
  transfer_out = 0;
  time_cpu = 0;
  
  unsigned long N = source_img.width * source_img.height;
  int32_t *output = (int32_t *)calloc(N, sizeof(int32_t));
  int32_t smallest, largest;
  {
    // Transfer phase
    std::string cpu_file = cpu_output_filename + ".pgm";
    pgm_image cpu_output_img;
    copy_pgm_image_size(&source_img, &cpu_output_img);

    // Start time
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    run_best_cpu(FILTER, FILTER_DIMENSION, source_img.matrix, cpu_output_img.matrix, source_img.width, source_img.height, &smallest, &largest);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_cpu, start, stop);

    // Save CPU generated image
    save_pgm_to_file(cpu_file.c_str(), &cpu_output_img);
    if (DEBUGMODE >= 2) {
      for (int i = 0; i < source_img.height; i++) {
        printf("[");
        for (int j = 0; j < source_img.width; j++) {
          printf("%d ", source_img.matrix[source_img.width*i + j]);
        }
        printf("]\n");
      }
      for (int i = 0; i < cpu_output_img.height; i++) {
        printf("[");
        for (int j = 0; j < cpu_output_img.width; j++) {
          printf("%d ", cpu_output_img.matrix[cpu_output_img.width*i + j]);
        }
        printf("]\n");
      }
    }

    // End phase
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    print_run(time_cpu, 0, computation_time, transfer_in, transfer_out);
    memcpy(output, cpu_output_img.matrix, N*sizeof(int32_t));
    destroy_pgm_image(&cpu_output_img);
  }

  /* GPU Implementations */
  // Reduction step
  /*
  const unsigned THREADS_PER_BLOCK = 512;
  int32_t *d_output = NULL;
  int32_t *d_min = NULL;
  int32_t *d_max = NULL;
  unsigned s = N;
  unsigned blocks = (s + 2*THREADS_PER_BLOCK - 1)/(2*THREADS_PER_BLOCK);
  cudaMalloc((void **)&d_output, sizeof(int32_t)*N);
  cudaMalloc((void **)&d_min, sizeof(int32_t)*blocks);
  cudaMalloc((void **)&d_max, sizeof(int32_t)*blocks);
  cudaMemcpy(&d_output, cpu_output_img.matrix, sizeof(int32_t)*N, cudaMemcpyHostToDevice);
  minmaxreduce<<<blocks, THREADS_PER_BLOCK, THREADS_PER_BLOCK*sizeof(int32_t)>>>(d_output, d_min, 1, N);
  minmaxreduce<<<blocks, THREADS_PER_BLOCK, THREADS_PER_BLOCK*sizeof(int32_t)>>>(d_output, d_max, -1, N);
  while (blocks > 1) {
    s = blocks;
    blocks = (s + 2*THREADS_PER_BLOCK - 1)/(2*THREADS_PER_BLOCK);
    minmaxreduce<<<blocks, THREADS_PER_BLOCK, THREADS_PER_BLOCK*sizeof(int32_t)>>>(d_min, d_min, 1, s);
    minmaxreduce<<<blocks, THREADS_PER_BLOCK, THREADS_PER_BLOCK*sizeof(int32_t)>>>(d_max, d_max, -1, s);
  }
  int32_t smallest, largest;
  cudaMemcpy(&omg, d_min, sizeof(int32_t), cudaMemcpyDeviceToHost);
  cudaMemcpy(&zomg, d_max, sizeof(int32_t), cudaMemcpyDeviceToHost);
  cudaFree(d_output);
  cudaFree(d_min);
  cudaFree(d_max);
  printf("{%d %d}\n", smallest, largest);
  */

  // Kernel computations
  for (int k = 1; k <= 5; k++)
  {
    computation_time = 0;
    transfer_in = 0;
    transfer_out = 0; 

    // Host Transfer phase
    std::string gpu_file = base_gpu_output_filename + "_G" + std::to_string(k) + ".pgm";
    int32_t *output_img_matrix = (int32_t *)calloc(N, sizeof(int32_t));    // Host
    int32_t *gpu_img_matrix = NULL;         // Device
    int32_t *gpu_output_img_matrix = NULL;  // Device
    int8_t *gpu_filter = NULL;              // Device
    cudaMalloc((void **)&gpu_output_img_matrix, N*sizeof(int32_t));
    cudaMalloc((void **)&gpu_img_matrix, N*sizeof(int32_t));
    cudaMalloc((void **)&gpu_filter, FILTER_DIMENSION*FILTER_DIMENSION*sizeof(int8_t));

    // Transfer to Device phase
    cudaEventCreate(&start); ///
    cudaEventCreate(&stop);  ///
    cudaEventRecord(start);  ///
    cudaMemcpy(gpu_img_matrix, source_img.matrix, N*sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_output_img_matrix, output_img_matrix, N*sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_filter, FILTER, FILTER_DIMENSION*FILTER_DIMENSION*sizeof(int8_t), cudaMemcpyHostToDevice);
    cudaEventRecord(stop);      ///
    cudaEventSynchronize(stop); /// 
    cudaEventElapsedTime(&transfer_in, start, stop); ///
    cudaEventDestroy(start);    ///
    cudaEventDestroy(stop);     ///

    // GPU Phase
    cudaEventCreate(&start); ///
    cudaEventCreate(&stop);  ///
    cudaEventRecord(start);  ///
    // Filtering step
    switch (k) {
      case 1: 
      	run_kernel1(gpu_filter, FILTER_DIMENSION, gpu_img_matrix, gpu_output_img_matrix, source_img.width, source_img.height, smallest, largest); 
        break;
      case 2: 
      	run_kernel2(gpu_filter, FILTER_DIMENSION, gpu_img_matrix, gpu_output_img_matrix, source_img.width, source_img.height, smallest, largest); 
        break;
      case 3: 
	      run_kernel3(gpu_filter, FILTER_DIMENSION, gpu_img_matrix, gpu_output_img_matrix, source_img.width, source_img.height, smallest, largest); 
        break;
      case 4: 
      	run_kernel4(gpu_filter, FILTER_DIMENSION, gpu_img_matrix, gpu_output_img_matrix, source_img.width, source_img.height, smallest, largest); 
        break;
      case 5: 
      	run_kernel5(gpu_filter, FILTER_DIMENSION, gpu_img_matrix, gpu_output_img_matrix, source_img.width, source_img.height, smallest, largest); 
        break;
    }

    cudaEventRecord(stop);      ///
    cudaEventSynchronize(stop); ///
    cudaEventElapsedTime(&computation_time, start, stop); ///
    cudaEventDestroy(start);    ///
    cudaEventDestroy(stop);     ///

    // Return phase
    cudaEventCreate(&start); ///
    cudaEventCreate(&stop);  ///
    cudaEventRecord(start);  ///
    cudaMemcpy(output_img_matrix, gpu_output_img_matrix, N*sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);      ///
    cudaEventSynchronize(stop); ///
    cudaEventElapsedTime(&transfer_out, start, stop); ///
    cudaEventDestroy(start);    ///
    cudaEventDestroy(stop);     ///

    // Print phase
    if (DEBUGMODE) {
      for (int i = 0; i < source_img.height; i++) {
        MATRIXPRINT("[");
        for (int j = 0; j < source_img.width; j++) {
          MATRIXPRINT("%d ", output_img_matrix[source_img.width*i + j]);
	  if (output_img_matrix[source_img.width*i + j] != output[source_img.width*i + j]) ERRORPRINT("%d:%d ", output_img_matrix[source_img.width*i + j], output[source_img.width*i + j]);
        }
        MATRIXPRINT("]\n");
      }
    }
    print_run(time_cpu, k, computation_time, transfer_in, transfer_out);

    // Save to output image
    pgm_image output_img;
    copy_pgm_image_size(&source_img, &output_img);
    output_img.matrix = output_img_matrix;
    if (save_pgm_to_file(gpu_file.c_str(), &output_img) != NO_ERR) {
      fprintf(stderr,"Failed to save to file.");
    }

    free(output_img_matrix);
    cudaFree(gpu_filter);
    cudaFree(gpu_img_matrix);
    cudaFree(gpu_output_img_matrix);
  }

  /* Repeat that for all 5 kernels. Don't hesitate to ask if you don't
   * understand the idea. */
  free(output);
}
