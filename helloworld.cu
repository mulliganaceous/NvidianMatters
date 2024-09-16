#include <stdio.h>
#define M 3
#define N 4

__global__ void helloCUDA(float f)
{
    printf("Hello thread %d,%d, f=%f\n", threadIdx.x, threadIdx.y, threadIdx.x*f);
}

int main()
{
    helloCUDA<<<M, N>>>(3.1415926535);
    cudaDeviceSynchronize();
    return 0;
}
