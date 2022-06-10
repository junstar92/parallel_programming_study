// 04_single-block-loop
#include <stdio.h>

__global__
void loop()
{
    printf("This is iteration number %d\n", threadIdx.x);
}

int main()
{
    loop<<<1, 10>>>();
    cudaDeviceSynchronize();
}