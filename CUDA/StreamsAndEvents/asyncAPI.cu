/*****************************************************************************
 * File:        asyncAPI.cu
 * Description: This is an example of using CUDA events to control asynchronous
 *              work launched on the GPU. In this example, asynchronous copies
 *              and an asynchronous kernel are used. A CUDA event is used to
 *              determine when that work has completed.
 *              
 * Compile:     nvcc -o asyncAPI asyncAPI.cu -I..
 * Run:         ./asyncAPI
 *****************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "common/common.h"

__global__
void kernel(float* g_data, float value)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    g_data[idx] = g_data[idx] + value;
}

bool checkResult(float* data, const int N, const float x)
{
    for (int i = 0; i < N; i++) {
        if (data[i] != x) {
            printf("Error! data[%d] = %f, ref = %f\n", i, data[i], x);
            return false;
        }
    }

    return true;
}

int main(int argc, char** argv)
{
    int dev = 0;
    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("> Using device %d: %s\n", dev, deviceProp.name);
    CUDA_CHECK(cudaSetDevice(dev));

    int num = 1 << 24;
    int nBytes = num * sizeof(float);
    float value = 10.0f;

    // allocate host memory
    float *h_a;
    CUDA_CHECK(cudaMallocHost((void**)&h_a, nBytes));
    memset(h_a, 0, nBytes);

    // allocate device memory
    float *d_a;
    CUDA_CHECK(cudaMalloc((void**)&d_a, nBytes));
    CUDA_CHECK(cudaMemset(d_a, 255, nBytes));

    // set kernel launch configuration
    dim3 block = dim3(512);
    dim3 grid = dim3((num + block.x - 1) / block.x);

    // create cuda event handles
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&stop));

    // asynchronously issue work to the GPU (all to stream 0)
    CUDA_CHECK(cudaMemcpyAsync(d_a, h_a, nBytes, cudaMemcpyHostToDevice));
    kernel<<<grid, block>>>(d_a, value);
    CUDA_CHECK(cudaMemcpyAsync(h_a, d_a, nBytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(stop));

    // have CPU do some work while waiting for stage 1 to finish
    unsigned long int counter = 0;
    while (cudaEventQuery(stop) == cudaErrorNotReady)
        counter++;
    
    // print the cpu and gpu times
    printf("CPU executed %lu iterations while waiting for GPU to finish\n", counter);

    // check the output for correctness
    bool results = checkResult(h_a, num, value);

    // release resources
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFreeHost(h_a));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaDeviceReset());

    return 0;
}