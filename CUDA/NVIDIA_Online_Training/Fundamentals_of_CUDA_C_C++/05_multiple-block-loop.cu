// 05_multiple-block-loop
#include <stdio.h>

__global__
void loop()
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    printf("This is iteration number %d\n", idx);
}

int main()
{
    loop<<<2, 5>>>();
    cudaDeviceSynchronize();
}

// CPU-only

int N = 2<<20;
size_t size = N * sizeof(int);

int *a;
a = (int *)malloc(size);

// Use `a` in CPU-only program.

free(a);
// Accelerated

int N = 2<<20;
size_t size = N * sizeof(int);

int *a;
// Note the address of `a` is passed as first argument.
cudaMallocManaged(&a, size);

// Use `a` on the CPU and/or on any GPU in the accelerated system.

cudaFree(a);