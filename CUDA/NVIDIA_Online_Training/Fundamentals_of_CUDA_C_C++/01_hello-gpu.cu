#include <stdio.h>

void helloCPU()
{
    printf("Hello from the CPU.\n");
}

__global__ void helloGPU()
{
    printf("Hello from the GPU.\n");
}

int main()
{
    helloCPU();

    helloGPU<<<1, 1>>>();
    cudaDeviceSynchronize();
}