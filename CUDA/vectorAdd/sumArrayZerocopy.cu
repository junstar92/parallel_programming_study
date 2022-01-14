/*****************************************************************************
 * File:        sumArrayZerocopy.cu
 * Description: This shows the use of zero-copy memory to remove the need to 
 *              explicitly issue a memcpy operation between the host and device.
 *              By mapping host, page-locked memory into the device's address space,
 *              the address can directly reference a host array and transfer
 *              its contents over the PCIe bus.
 *              
 * Compile:     nvcc -O3 -o sumArrayZerocopy sumArrayZerocopy.cu -I..
 * Run:         ./sumArrayZerocopy <n>
 *                  <n> : the power for the number of elements in array
 *                        -> the number of elements: 2^n
 *****************************************************************************/
#include <stdio.h>
#include <cuda_runtime.h>
#include "common/common.h"

void checkResult(float* hostRef, float* gpuRef, const int N)
{
    double epsilon = 1.0e-8;

    for (int i = 0; i < N; i++) {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon) {
            printf("Arrays do not match!\n");
            printf("host %5.2f gpu %5.2f at current %d\n", hostRef[i], gpuRef[i], i);
            break;
        }
    }
}

void initialData(float* arr, const int N)
{
    for (int i = 0; i < N; i++)
        arr[i] = (float)(rand() & 0xFF) / 10.0f;
}

void sumArraysOnHost(float* A, float* B, float* C, const int N)
{
    for (int i = 0; i < N; i++)
        C[i] = A[i] + B[i];
}

__global__
void sumArrays(float* A, float* B, float* C, const int N)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (idx < N)
        C[idx] = A[idx] + B[idx];
}

__global__
void sumArraysZeroCopy(float* A, float* B, float* C, const int N)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (idx < N)
        C[idx] = A[idx] + B[idx];
}

int main(int argc, char** argv)
{
    // setup device
    int dev = 0;
    CUDA_CHECK(cudaSetDevice(dev));

    // get device properties
    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, dev));

    // check if support mapped memory
    if (!deviceProp.canMapHostMemory) {
        printf("Device %d does not support mapping CPU host memory!\n", dev);
        CUDA_CHECK(cudaDeviceReset());
        return -1;
    }

    printf("Using Device %d: %s\n", dev, deviceProp.name);

    // setup data size of arrays
    int power = 10;
    if (argc > 1)
        power = atoi(argv[1]);
    
    int nElem = 1 << power;
    size_t nBytes = nElem * sizeof(float);

    if (power < 18) {
        printf("Array size %d power %d  nBytes %3.0f KB\n", nElem, power, nBytes/1024.f);
    }
    else {
        printf("Array size %d power %d  nBytes %3.0f MB\n", nElem, power, nBytes/(1024.f * 1024.f));
    }

    // part 1: using device memory
    float* h_A, *h_B, *hostRef, *gpuRef;
    h_A = (float*)malloc(nBytes);
    h_B = (float*)malloc(nBytes);
    hostRef = (float*)malloc(nBytes);
    gpuRef = (float*)malloc(nBytes);

    // initialize data at host side
    initialData(h_A, nElem);
    initialData(h_B, nElem);
    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    // add vector at host side for result checks
    sumArraysOnHost(h_A, h_B, hostRef, nElem);

    // malloc device global memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void**)&d_A, nBytes));
    CUDA_CHECK(cudaMalloc((void**)&d_B, nBytes));
    CUDA_CHECK(cudaMalloc((void**)&d_C, nBytes));

    // transfer data from host to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice));

    // setup execution configuration
    int threads = 512;
    dim3 blocks(threads);
    dim3 grids((blocks.x + nElem - 1) / blocks.x);

    sumArrays<<<grids, blocks>>>(d_A, d_B, d_C, nElem);

    // copy kernel result back to host side
    CUDA_CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    // check device results
    checkResult(hostRef, gpuRef, nElem);

    // free device global memory
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    
    // free host memory
    free(h_A);
    free(h_B);

    // part 2: using zero-copy memory for array A and B
    // allocate zero-copy memory
    CUDA_CHECK(cudaHostAlloc((void**)&h_A, nBytes, cudaHostAllocMapped));
    CUDA_CHECK(cudaHostAlloc((void**)&h_B, nBytes, cudaHostAllocMapped));

    // initialize data at host side
    initialData(h_A, nElem);
    initialData(h_B, nElem);
    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    // pass the pointer to device
    CUDA_CHECK(cudaHostGetDevicePointer((void**)&d_A, (void*)h_A, 0));
    CUDA_CHECK(cudaHostGetDevicePointer((void**)&d_B, (void*)h_B, 0));
    
    // add vector at host side for result checks
    sumArraysOnHost(h_A, h_B, hostRef, nElem);

    // execute kernel with zero copy memory
    sumArraysZeroCopy<<<grids, blocks>>>(d_A, d_B, d_C, nElem);

    // copy kernel result back to host side
    CUDA_CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    // check device results
    checkResult(hostRef, gpuRef, nElem);

    // free device global memory
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFreeHost(h_A));
    CUDA_CHECK(cudaFreeHost(h_B));
    
    // free host memory
    free(hostRef);
    free(gpuRef);

    // reset device
    CUDA_CHECK(cudaDeviceReset());
    return 0;
}