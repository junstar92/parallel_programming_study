/*****************************************************************************
 * File:        AoS.cu
 * Description: This is a simple example of using an array of structures to
 *              store data on the device.
 *              
 * Compile:     nvcc -O3 -o AoS AoS.cu -I..
 * Run:         ./AoS [n]
 *                  [n] : the number of threads in a block
 *****************************************************************************/

#include <stdio.h>
#include <cuda_runtime.h>
#include "common/common.h"

#define LEN 1 << 20

struct innerStruct {
    float x;
    float y;
};

void initialInnerStruct(innerStruct* in, const int N)
{
    for (int i = 0; i < N; i++) {
        in[i].x = (rand() & 0xFF) / 100.f;
        in[i].y = (rand() & 0xFF) / 100.f;
    }
}

void testInnerStructHost(innerStruct* data, innerStruct* result, const int N)
{
    for (int i = 0; i < N; i++) {
        result[i].x = data[i].x + 10.f;
        result[i].y = data[i].y + 20.f;
    }
}

void checkInnerStruct(innerStruct* hostRef, innerStruct* gpuRef, const int N)
{
    double epsilon = 1.0e-8;
    
    for (int i = 0; i < N; i++) {
        if (abs(hostRef[i].x - gpuRef[i].x) > epsilon) {
            printf("different on %dth element: host %f gpu %f\n", i, hostRef[i].x, gpuRef[i].x);
            printf("Arrays do not match.\n\n");

            break;
        }
        if (abs(hostRef[i].y - gpuRef[i].y) > epsilon) {
            printf("different on %dth element: host %f gpu %f\n", i, hostRef[i].y, gpuRef[i].y);
            printf("Arrays do not match.\n\n");

            break;
        }
    }
}

__global__
void testInnerStruct(innerStruct* data, innerStruct* result, const int N)
{
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < N) {
        innerStruct tmp = data[idx];
        tmp.x += 10.f;
        tmp.y += 20.f;
        result[idx] = tmp;
    }
}

__global__
void warmup(innerStruct* data, innerStruct* result, const int N)
{
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < N) {
        innerStruct tmp = data[idx];
        tmp.x += 10.f;
        tmp.y += 20.f;
        result[idx] = tmp;
    }
}

int main(int argc, char** argv)
{
    // setup device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Test struct of array at device %d: %s\n", dev, deviceProp.name);

    // allocate host memory
    int nElem = LEN;
    size_t nBytes = nElem * sizeof(innerStruct);
    innerStruct *h_A = (innerStruct*)malloc(nBytes);
    innerStruct *hostRef = (innerStruct*)malloc(nBytes);
    innerStruct *gpuRef = (innerStruct*)malloc(nBytes);

    // initialize host array
    initialInnerStruct(h_A, nElem);
    testInnerStructHost(h_A, hostRef, nElem);

    // allocate device memory
    innerStruct* d_A, *d_C;
    CUDA_CHECK(cudaMalloc((void**)&d_A, nBytes));
    CUDA_CHECK(cudaMalloc((void**)&d_C, nBytes));

    // copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));

    // setup execution configuration
    int threads = 128;
    if (argc > 1)
        threads = atoi(argv[1]);
    
    dim3 blocks(threads, 1);
    dim3 grids((blocks.x + nElem - 1) / blocks.x, 1);

    double start, finish;
    // kernel 1: warmup
    GET_TIME(start);
    warmup<<<grids, blocks>>>(d_A, d_C, nElem);
    cudaDeviceSynchronize();
    GET_TIME(finish);
    //printf("warpup      <<< %3d, %3d >>> elapsed %f sec\n", grids.x, blocks.x, finish-start);
    CUDA_CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
    checkInnerStruct(hostRef, gpuRef, nElem);

    // kernel 2: testInnerStruct
    GET_TIME(start);
    testInnerStruct<<<grids, blocks>>>(d_A, d_C, nElem);
    CUDA_CHECK(cudaDeviceSynchronize());
    GET_TIME(finish);
    printf("innerstruct <<< %3d, %3d >>> elapsed %f sec\n", grids.x, blocks.x, finish-start);
    CUDA_CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
    checkInnerStruct(hostRef, gpuRef, nElem);

    // free memories bost host and device
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A);
    free(hostRef);
    free(gpuRef);

    CUDA_CHECK(cudaDeviceReset());
    return 0;
}