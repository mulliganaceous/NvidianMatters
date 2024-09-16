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
#include <iostream>

#include "kernels.h"
#include "pgm.h"
#include "gtest/gtest.h"

// This file is assuming you are using the 9x9 filter. If you're not,
// change it.
const int8_t FILTER[] = {
    0, 1, 1, 2, 2, 2,   1,   1,   0, 1, 2, 4, 5, 5,   5,   4,   2,
    1, 1, 4, 5, 3, 0,   3,   5,   4, 1, 2, 5, 3, -12, -24, -12, 3,
    5, 2, 2, 5, 0, -24, -40, -24, 0, 5, 2, 2, 5, 3,   -12, -24, -12,
    3, 5, 2, 1, 4, 5,   3,   0,   3, 5, 4, 1, 1, 2,   4,   5,   5,
    5, 4, 2, 1, 0, 1,   1,   2,   2, 2, 1, 1, 0,
};
const int FILTER_DIMENSION = 9;

// ============================================================================

// void compare_kernel_against_handwritten_example_1(int kernel) {
//   const int width = 2;
//   const int height = 2;

//   int32_t image[] = {0, 1, 2, 3};
//   pgm_image source{width, height, 255, image};

//   pgm_image output_img;
//   copy_pgm_image_size(&source, &output_img);
//   int32_t smallest, largest;

//   switch (kernel) {
//   case 0: {
//     run_best_cpu(FILTER, FILTER_DIMENSION, image, output_img.matrix, width,
//                  height, &smallest, &largest);
//     break;
//   }
//   case 1: {
//     run_kernel1(FILTER, FILTER_DIMENSION, image, output_img.matrix, width,
//                 height, smallest, largest);
//     break;
//   }
//   case 2: {
//     run_kernel2(FILTER, FILTER_DIMENSION, image, output_img.matrix, width,
//                 height, smallest, largest);
//     break;
//   }
//   case 3: {
//     run_kernel3(FILTER, FILTER_DIMENSION, image, output_img.matrix, width,
//                 height, smallest, largest);
//     break;
//   }
//   case 4: {
//     run_kernel4(FILTER, FILTER_DIMENSION, image, output_img.matrix, width,
//                 height, smallest, largest);
//     break;
//   }
//   case 5: {
//     run_kernel5(FILTER, FILTER_DIMENSION, image, output_img.matrix, width,
//                 height, smallest, largest);
//     break;
//   }
//   }

//   int32_t expected[] = {255, 170, 85, 0};
//   for (int i = 0; i < width * height; i++) {
//     ASSERT_EQ(output_img.matrix[i], expected[i]) << i;
//   }
// }

// TEST(csc367a4, kernel0_handwritten_tests) {
//   compare_kernel_against_handwritten_example_1(0);
// }
// TEST(csc367a4, kernel1_handwritten_tests) {
//   compare_kernel_against_handwritten_example_1(1);
// }
// TEST(csc367a4, kernel2_handwritten_tests) {
//   compare_kernel_against_handwritten_example_1(2);
// }
// TEST(csc367a4, kernel3_handwritten_tests) {
//   compare_kernel_against_handwritten_example_1(3);
// }
// TEST(csc367a4, kernel4_handwritten_tests) {
//   compare_kernel_against_handwritten_example_1(4);
// }
// TEST(csc367a4, kernel5_handwritten_tests) {
//   compare_kernel_against_handwritten_example_1(5);
// }

// TEST(csc367a4, compare_kernels_against_each_other) {
//   const int WIDTH_MAX = 13;
//   const int HEIGHT_MAX = 19;
//   const int NUM_KERNELS = 6;

//   for (int width = 1; width < WIDTH_MAX; ++width) {
//     for (int height = 1; height < HEIGHT_MAX; ++height) {

//       pgm_image source;
//       create_random_pgm_image(&source, width, height);

//       pgm_image outputs[NUM_KERNELS];
//       for (int current_kernel = 0; current_kernel < NUM_KERNELS;
//            ++current_kernel) {
//         copy_pgm_image_size(&source, &outputs[current_kernel]);
//       }
//       int32_t smallest, largest;
//       run_best_cpu(FILTER, FILTER_DIMENSION, source.matrix, outputs[0].matrix, 
//                    width, height, &smallest, &largest);
//       run_kernel1(FILTER, FILTER_DIMENSION, source.matrix, outputs[1].matrix,
//                    width, height, smallest, largest);
//       run_kernel2(FILTER, FILTER_DIMENSION, source.matrix, outputs[2].matrix,
//                    width, height, smallest, largest);
//       run_kernel3(FILTER, FILTER_DIMENSION, source.matrix, outputs[3].matrix,
//                    width, height, smallest, largest);
//       run_kernel4(FILTER, FILTER_DIMENSION, source.matrix, outputs[4].matrix,
//                    width, height, smallest, largest);
//       run_kernel5(FILTER, FILTER_DIMENSION, source.matrix, outputs[5].matrix,
//                    width, height, smallest, largest);

//       for (int current_kernel = 1; current_kernel < NUM_KERNELS;
//            ++current_kernel) {
//         for (int pixel = 0; pixel < width * height; pixel++) {
//           ASSERT_EQ(outputs[current_kernel].matrix[pixel],
//                     outputs[0].matrix[pixel])
//               << "current_kernel = " << current_kernel << "\npixel = " << pixel
//               << "\na matrix with width = " << width << " height = " << height;
//         }
//       }

//       for (int current_kernel = 0; current_kernel < NUM_KERNELS;
//            ++current_kernel) {
//         destroy_pgm_image(&outputs[current_kernel]);
//       }
//     }
//   }
// }

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
