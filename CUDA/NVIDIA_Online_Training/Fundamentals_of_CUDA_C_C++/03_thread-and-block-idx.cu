// 03_thread-and-block-idx.cu
#include <stdio.h>

__global__
void printSuccessForCorrectExecutionConfiguration()
{
    if (threadIdx.x == 1023 && blockIdx.x == 255) {
        printf("Success.\n");
    }
}

int main()
{
    printSuccessForCorrectExecutionConfiguration<<<256, 1024>>>();
    cudaDeviceSynchronize();
}