/*****************************************************************************
 * File:        my-atomic-add.cu
 * Description: This is an example to illustrates implementation of custom atomic
 *              operations using CUDA's build-in atomicCAS function to implement
 *              atomic signed 32-bit integer addition
 *              
 * Compile:     nvcc -o my-atomic-add my-atomic-add.cu -I..
 * Run:         ./my-atomic-add
 *****************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "common/common.h"

__device__
int myAtomicAdd(int* address, int incr)
{
    // Create an initial guess for the value stored at *address
    int guess = *address;
    int oldValue = atomicCAS(address, guess, guess + incr);

    // Loop while the quess is incorrect
    while (oldValue != guess) {
        guess = oldValue;
        oldValue = atomicCAS(address, guess, guess + incr);
    }
    
    return oldValue;
}

__global__
void kernel(int *sharedInteger)
{
    myAtomicAdd(sharedInteger, 1);
}

int main(int argc, char **argv)
{
    int h_sharedInteger;
    int *d_sharedInteger;
    CUDA_CHECK(cudaMalloc((void **)&d_sharedInteger, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_sharedInteger, 0x00, sizeof(int)));

    kernel<<<4, 128>>>(d_sharedInteger);

    CUDA_CHECK(cudaMemcpy(&h_sharedInteger, d_sharedInteger, sizeof(int), cudaMemcpyDeviceToHost));
    printf("4 x 128 increments led to value of %d\n", h_sharedInteger);

    return 0;
}