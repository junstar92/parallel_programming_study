/*****************************************************************************
 * File:        SoA.cu
 * Description: This is a simple example of using a structure of arrays to
 *              store data on the device.
 *              
 * Compile:     nvcc -O3 -o SoA SoA.cu -I..
 * Run:         ./SoA [n]
 *                  [n] : the number of threads in a block
 *****************************************************************************/

#include <stdio.h>
#include <cuda_runtime.h>
#include "common/common.h"

#define LEN 1 << 20

struct innerArray {
    float x[LEN];
    float y[LEN];
};

void initialInnerArray(innerArray* in, const int N)
{
    for (int i = 0; i < N; i++) {
        in->x[i] = (rand() & 0xFF) / 100.f;
        in->y[i] = (rand() & 0xFF) / 100.f;
    }
}

void testInnerArrayHost(innerArray* data, innerArray* result, const int N)
{
    for (int i = 0; i < N; i++) {
        result->x[i] = data->x[i] + 10.f;
        result->y[i] = data->y[i] + 20.f;
    }
}

void checkInnerArray(innerArray* hostRef, innerArray* gpuRef, const int N)
{
    double epsilon = 1.0e-8;
    
    for (int i = 0; i < N; i++) {
        if (abs(hostRef->x[i] - gpuRef->x[i]) > epsilon) {
            printf("different on %dth element: host %f gpu %f\n", i, hostRef->x[i], gpuRef->x[i]);
            printf("Arrays do not match.\n\n");

            break;
        }
        if (abs(hostRef->y[i] - gpuRef->y[i]) > epsilon) {
            printf("different on %dth element: host %f gpu %f\n", i, hostRef->y[i], gpuRef->y[i]);
            printf("Arrays do not match.\n\n");

            break;
        }
    }
}

__global__
void testInnerArray(innerArray* data, innerArray* result, const int N)
{
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < N) {
        float tmpX = data->x[idx];
        float tmpY = data->y[idx];

        tmpX += 10.f;
        tmpY += 20.f;
        result->x[idx] = tmpX;
        result->y[idx] = tmpY;
    }
}

__global__
void warmup(innerArray* data, innerArray* result, const int N)
{
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < N) {
        float tmpX = data->x[idx];
        float tmpY = data->y[idx];

        tmpX += 10.f;
        tmpY += 20.f;
        result->x[idx] = tmpX;
        result->y[idx] = tmpY;
    }
}

int main(int argc, char** argv)
{
    // setup device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Test struct of array at device %d: %s\n", dev, deviceProp.name);
    CUDA_CHECK(cudaSetDevice(dev));

    // allocate host memory
    int nElem = LEN;
    size_t nBytes = sizeof(innerArray);
    innerArray *h_A = (innerArray*)malloc(nBytes);
    innerArray *hostRef = (innerArray*)malloc(nBytes);
    innerArray *gpuRef = (innerArray*)malloc(nBytes);

    // initialize host array
    initialInnerArray(h_A, nElem);
    testInnerArrayHost(h_A, hostRef, nElem);

    // allocate device memory
    innerArray* d_A, *d_C;
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
    checkInnerArray(hostRef, gpuRef, nElem);

    // kernel 2: testInnerArray
    GET_TIME(start);
    testInnerArray<<<grids, blocks>>>(d_A, d_C, nElem);
    CUDA_CHECK(cudaDeviceSynchronize());
    GET_TIME(finish);
    printf("innerarray  <<< %3d, %3d >>> elapsed %f sec\n", grids.x, blocks.x, finish-start);
    CUDA_CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
    checkInnerArray(hostRef, gpuRef, nElem);

    // free memories bost host and device
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A);
    free(hostRef);
    free(gpuRef);

    CUDA_CHECK(cudaDeviceReset());
    return 0;
}