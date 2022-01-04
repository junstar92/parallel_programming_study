/*****************************************************************************
 * File:        simpleDivergence.cu
 * Description: Measure the performance of some kernels.
 *              One has warp divergence and others doesn't have warp divergence.
 *              
 * Compile:     nvcc -g -G -arch=sm_75 -o simpleDivergence simpleDivergence.cu -I..
 * Run:         ./simpleDivergence
 * Argument:    n.a
 *****************************************************************************/
#include <stdio.h>
#include <cuda_runtime.h>
#include <common/common.h>

__global__ void mathKernel1(float* c)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
    float a, b;
    a = b = 0.0f;
    
    if (tid % 2 == 0) {
    	a = 100.0f;
    }
    else {
    	b = 200.0f;
    }
    c[tid] = a + b;
}

__global__ void mathKernel2(float* c)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
    float a, b;
    a = b = 0.0f;
    
    if ((tid / warpSize) % 2 == 0) {
    	a = 100.0f;
    }
    else {
    	b = 200.0f;
    }
    c[tid] = a + b;
}

__global__ void mathKernel3(float* c)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
    float a, b;
    a = b = 0.0f;
    
    bool pred = (tid % 2 == 0);

    if (pred) {
    	a = 100.0f;
    }

    if (!pred) {
    	b = 200.0f;
    }

    c[tid] = a + b;
}

__global__ void mathKernel4(float* c)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
    float a, b;
    a = b = 0.0f;
    
    int itid = tid >> 5;

    if (itid & 0x01 == 0) {
    	a = 100.0f;
    }
    else {
    	b = 200.0f;
    }
    
    c[tid] = a + b;
}

__global__ void warmingup(float *c)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    float a, b;
    a = b = 0.0f;
    
    if ((tid / warpSize) % 2 == 0) {
    	a = 100.0f;
    }
    else {
    	b = 200.0f;
    }
    c[tid] = a + b;
}

int main(int argc, char** argv)
{
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("%s using Device %d: %s\n", argv[0], dev, deviceProp.name);

    // set up data size
    int size = 64;
    int blockSize = 64;
    if (argc > 1)
        blockSize = atoi(argv[1]);
    if (argc > 2)
        size = atoi(argv[2]);
    printf("Data size: %d ", size);

    // set up execution configuration
    dim3 block(blockSize, 1);
    dim3 grid((size+block.x-1) / block.x, 1);
    printf("Excution Configure (block %d grid %d)\n", block.x, grid.x);

    // allocate gpu memory
    float *d_C;
    size_t nBytes = size * sizeof(float);
    cudaMalloc((void**)&d_C, nBytes);

    double start, finish;
    // run a warmup kernel to remove overhead
    cudaDeviceSynchronize();
    GET_TIME(start);
    warmingup<<<grid, block>>>(d_C);
    cudaDeviceSynchronize();
    GET_TIME(finish);
    printf("warmup      <<< %4d %4d >>> elapsed %f sec\n", grid.x, block.x, finish-start);

    // run kernel 1
    GET_TIME(start);
    mathKernel1<<<grid, block>>>(d_C);
    cudaDeviceSynchronize();
    GET_TIME(finish);
    printf("mathKernel1 <<< %4d %4d >>> elapsed %f sec\n", grid.x, block.x, finish-start);

    // run kernel 2
    GET_TIME(start);
    mathKernel2<<<grid, block>>>(d_C);
    cudaDeviceSynchronize();
    GET_TIME(finish);
    printf("mathKernel2 <<< %4d %4d >>> elapsed %f sec\n", grid.x, block.x, finish-start);

    // run kernel 3
    GET_TIME(start);
    mathKernel3<<<grid, block>>>(d_C);
    cudaDeviceSynchronize();
    GET_TIME(finish);
    printf("mathKernel3 <<< %4d %4d >>> elapsed %f sec\n", grid.x, block.x, finish-start);

    // run kernel 4
    GET_TIME(start);
    mathKernel4<<<grid, block>>>(d_C);
    cudaDeviceSynchronize();
    GET_TIME(finish);
    printf("mathKernel4 <<< %4d %4d >>> elapsed %f sec\n", grid.x, block.x, finish-start);


    // free gpu memory and reset device
    cudaFree(d_C);
    cudaDeviceReset();

    return 0;
}