cmake_minimum_required(VERSION 3.10)

# Do no set the host compiler to a version higher than the one supported by
# nvcc:
# https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html
# This may be unnecessary and has to be updated every term based on the
# the gcc version of the lab / local machine.
# set(CMAKE_CXX_COMPILER "g++-8")

project(csc367a4 LANGUAGES CXX CUDA)

add_subdirectory(googletest)

set(CMAKE_CUDA_FLAGS "-arch=sm_61")

add_library(a4_options INTERFACE)
target_compile_features(a4_options INTERFACE cxx_std_11)

add_library(a4_kernels
  pgm.cu
  best_cpu.cu
  kernel1.cu
  kernel2.cu
  kernel3.cu
  kernel4.cu
  kernel5.cu
)

target_link_libraries(a4_kernels
  PUBLIC a4_options
)

add_executable(main
  main.cu
)

target_link_libraries(main
  PUBLIC a4_kernels a4_options
)

add_library(pgm
  pgm.cu
)

add_executable(pgm_creator
  pgm_creator.cu
)

target_link_libraries(pgm_creator
  PUBLIC pgm a4_options
)

add_executable(test_solution
  tests.cu
)

target_link_libraries(test_solution
  PUBLIC pgm a4_kernels gtest a4_options
)
