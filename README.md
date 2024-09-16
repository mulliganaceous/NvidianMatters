# Nvidian matters

All Nvidian programming and demos goes there.

* A background (5m)
        * Prerequisite knowledge
        * What you will get out of it
        * Who am I
* Introduction to Nvidia (5m)
* Introduction to CUDA and GPGPU (5m)
        * GPGPU (2m)
        * Use cases (3m)
* CUDA Terms walthrough (20m)
        * Compilation and environment (hello world)
        * Deviceside, hostside, Kernel (dot product)
        * blocks and grids (matrix multiplication, convolution)
* Demo One (15m)
        * The Riemann zeta function
        * Graphing the Riemann zeta function (feat. opengl)
* Demo Two (5m)
        * Tensorflow GPU acceleration
        * CUDNN and cuQuantum
* Outro (AET)
        * Who am I
        * Later goals

# Introduction

## Prerequisites

An interest in mathematical and computational sciences, and or an interest in artificial intelligence.
C and C++ knowledge, basic compiler commands. Get tubular!

For Nvidian-specific matters, you should have a computer with a Nvidia GPU (such as a gaming computer), and following the instructions on installing Nvidia drivers and the CUDA toolkit. We will also very briefly look at Nvidia cuDNN and TensorRT. 

Unfortunately, Apple has declined to support the use of Nvidian cards since 2019. If you do not have a Nvidian laptop, try Google Colab instead and run on the cloud, and go with the T4!

### What you will get out of it

You will be able to massively parallelize operations that involve very large vectors and matrices, the hallmark of graphics processing and deep learning.

## What is Nvidia?

Nvidia is a cross between _invidia_, _video_, and _next_. Coined thirty-one years ago in Sunnyvale, California from a former AMD, Sun, and IBM employee, one day at a Denny's roadside diner.

The term "GPU" is not of Nvidian origin. It was actually coined by Sony to refer to the Toshiba-designed Sony GPU for the PSX in 1994. Nvidia is now the leading supplier of AI hardware and software, because AI has surprisingly many computational elements and characteristics in common with graphics processing.

## What is CUDA?

A barracuda?

### Libraries

Many libraries within CUDA, including:
* cuBLAS
* cuFFT
* cuRAND
* cuSOLVER
* cuSPARSE
* cuTENSOR
* NPP
* nvJPEG
* nvTIFF
* cuSOLVERMp
* GPUDirect

# Cuda Codealong

# Do you want a million dollars?

# Looking at defects

# Outroduction

Flutter and Dart developer, building a TTS application, then a Quantum algorithms developer. Worked at MTO before. Almost got into Intel.
From the University of Toronto. Admitted to PennApps since 2019, also my very first hackathon. But did not win a prize.

* Nvidia AI Summit
        * Washington, DC
        * October 7-9, 2024
* Grace Hopper 2024
        * Philadelphia, PA (hybrid)
        * Pennsylvania Convention Center
* Supercomputing 2024
        * Atlanta, GA
        * More details coming soon…

If you have an AMD Radeon or Macbook, try ROCm.