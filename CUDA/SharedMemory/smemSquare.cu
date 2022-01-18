/*****************************************************************************
 * File:        smemSquare.cu
 * Description: This is an example of using shared memory to transpose square
 *              thread coordinates of a CUDA grid into a global memory array.
 *              Different kernels below demonstrae performing reads and writes
 *              with different ordering, as optimizing using memory padding.
 *              
 * Compile:     nvcc -o smemSquare smemSquare.cu -I..
 * Run:         ./smemSquare
 *****************************************************************************/
#include <stdio.h>
#include <cuda_runtime.h>
#include "common/common.h"

#define BDIMX 32
#define BDIMY 32
#define IPAD 1

__global__
void setRowReadRow(int* out)
{
    // static shared memory
    __shared__ int tile[BDIMY][BDIMX];

    // mapping from thread index to global memory index
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

    // shared memory store operation
    tile[threadIdx.y][threadIdx.x] = idx;

    // wait for all threads to complete
    __syncthreads();

    // shared memory load operation
    out[idx] = tile[threadIdx.y][threadIdx.x];
}

__global__
void setColReadCol(int* out)
{
    // static shared memory
    __shared__ int tile[BDIMX][BDIMY];

    // mapping from thread index to global memory index
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

    // shared memory store operation
    tile[threadIdx.x][threadIdx.y] = idx;

    // wait for all threads to complete
    __syncthreads();

    // shared memory load operation
    out[idx] = tile[threadIdx.x][threadIdx.y];
}

int main(int argc, char** argv)
{
    // setup device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Starting at device %d: %s\n", dev, deviceProp.name);
    CUDA_CHECK(cudaSetDevice(dev));

    cudaSharedMemConfig pConfig;
    CUDA_CHECK(cudaDeviceGetSharedMemConfig ( &pConfig ));
    printf("with Bank Mode:%s \n", pConfig == 1 ? "4-Byte" : "8-Byte");

    // set up array size 2048
    int nx = BDIMX;
    int ny = BDIMY;

    size_t nBytes = nx * ny * sizeof(int);

    // execution configuration
    dim3 block (BDIMX, BDIMY);
    dim3 grid  (1, 1);
    printf("<<< grid (%d,%d) block (%d,%d)>>>\n", grid.x, grid.y, block.x, block.y);

    // allocate device memory
    int *d_C;
    CUDA_CHECK(cudaMalloc((void**)&d_C, nBytes));
    int *gpuRef  = (int *)malloc(nBytes);

    CUDA_CHECK(cudaMemset(d_C, 0, nBytes));
    setColReadCol<<<grid, block>>>(d_C);
    CUDA_CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemset(d_C, 0, nBytes));
    setRowReadRow<<<grid, block>>>(d_C);
    CUDA_CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    // free host and device memory
    CUDA_CHECK(cudaFree(d_C));
    free(gpuRef);

    // reset device
    CUDA_CHECK(cudaDeviceReset());
    return 0;
}