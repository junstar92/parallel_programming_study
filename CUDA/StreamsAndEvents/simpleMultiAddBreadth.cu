/*****************************************************************************
 * File:        simpleMultiAddBreadth.cu
 * Description: This is an example to demonstrate overlapping computation and 
 *              communication by partitioning a data set asynchronously launching
 *              the memory copies and kernels for each subset. Launching all
 *              transfers and kernels for a given subset in the same CUDA stream
 *              ensures that computation on the device is not started until the
 *              necessary data has been transferred. However, because the work of
 *              each subset is independent of all other subsets, the communication
 *              and computation of different subsets will overlap.
 *              (This example launches copies and kernels in breadth-first order)
 *              
 * Compile:     nvcc -o simpleMultiAddBreadth simpleMultiAddBreadth.cu -I..
 * Run:         ./simpleMultiAddBreadth
 *****************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "common/common.h"

#define NSTREAM 4
#define NREPEAT 100
#define BDIM 128

void initialData(float* in, const int size)
{
    for (int i = 0; i < size; i++)
        in[i] = (rand() % 0xff) / 10.f;
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

    if (idx < N) {
        for (int i = 0; i < NREPEAT; i++)
            C[idx] = A[idx] + B[idx];
    }
}

void checkResult(float* hostRef, float* gpuRef, const int N)
{
    double epsilon = 1.e-8;

    for (int i = 0; i < N; i++) {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon) {
            printf("Arrays do not match!\n");
            printf("host %5.2f gpu %5.2f at %d\n", hostRef[i], gpuRef[i], i);
            break;
        }
    }
}

int main(int argc, char** argv)
{
    int dev = 0;
    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("> Using Device %d: %s\n", dev, deviceProp.name);
    CUDA_CHECK(cudaSetDevice(dev));

    // check if device support hyper-Q
    if (deviceProp.major < 3 || (deviceProp.major == 3 && deviceProp.minor < 5)) {
        if (deviceProp.concurrentKernels == 0) {
            printf("> GPU does not support concurrent kernel execution (SM 3.5 or higher required)\n");
            printf("> CUDA kernel runs will be serialized\n");
        }
        else {
            printf("> GPU does not support HyperQ\n");
            printf("> CUDA kernel runs will have limited concurrency\n");
        }
    }

    printf("> Compute Capability %d.%d hardware with %d multi-processors\n", deviceProp.major,
            deviceProp.minor, deviceProp.multiProcessorCount);
    
    // set up max connection
    char* iname = "CUDA_DEVICE_MAX_CONNECTIONS";
    _putenv_s(iname, "1");
    char* ivalue = getenv(iname);
    printf("> %s = %s\n", iname, ivalue);
    printf("> with streams = %d\n", NSTREAM);

    // set up data size of vectors
    int nElem = 1 << 18;
    printf("> vector size = %d\n", nElem);
    size_t nBytes = nElem * sizeof(float);

    // malloc pinned host memory for async memcpy
    float *h_A, *h_B, *hostRef, *gpuRef;
    CUDA_CHECK(cudaHostAlloc((void**)&h_A, nBytes, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc((void**)&h_B, nBytes, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc((void**)&gpuRef, nBytes, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc((void**)&hostRef, nBytes, cudaHostAllocDefault));

    // initialize data at host side
    initialData(h_A, nElem);
    initialData(h_B, nElem);
    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    // add vector at host side for result checks
    sumArraysOnHost(h_A, h_B, hostRef, nElem);

    // malloc device global memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((float**)&d_A, nBytes));
    CUDA_CHECK(cudaMalloc((float**)&d_B, nBytes));
    CUDA_CHECK(cudaMalloc((float**)&d_C, nBytes));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // invoke kernel at host side
    dim3 block(BDIM);
    dim3 grid ((nElem + block.x - 1) / block.x);
    printf("> grid (%d,%d) block (%d,%d)\n", grid.x, grid.y, block.x, block.y);

    // sequential operation
    CUDA_CHECK(cudaEventRecord(start, 0));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float memcpy_h2d_time;
    CUDA_CHECK(cudaEventElapsedTime(&memcpy_h2d_time, start, stop));

    CUDA_CHECK(cudaEventRecord(start, 0));
    sumArrays<<<grid, block>>>(d_A, d_B, d_C, nElem);
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float kernel_time;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_time, start, stop));

    CUDA_CHECK(cudaEventRecord(start, 0));
    CUDA_CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float memcpy_d2h_time;
    CUDA_CHECK(cudaEventElapsedTime(&memcpy_d2h_time, start, stop));
    float itotal = kernel_time + memcpy_h2d_time + memcpy_d2h_time;

    printf("\n");
    printf("Measured timings (throughput):\n");
    printf(" Memcpy host to device\t: %f ms (%f GB/s)\n",
           memcpy_h2d_time, (nBytes * 1e-6) / memcpy_h2d_time);
    printf(" Memcpy device to host\t: %f ms (%f GB/s)\n",
           memcpy_d2h_time, (nBytes * 1e-6) / memcpy_d2h_time);
    printf(" Kernel\t\t\t: %f ms (%f GB/s)\n",
           kernel_time, (nBytes * 2e-6) / kernel_time);
    printf(" Total\t\t\t: %f ms (%f GB/s)\n",
           itotal, (nBytes * 2e-6) / itotal);
        
    // grid parallel operation
    int iElem = nElem / NSTREAM;
    size_t iBytes = iElem * sizeof(float);
    grid.x = (iElem + block.x - 1) / block.x;

    cudaStream_t stream[NSTREAM];
    for (int i = 0; i < NSTREAM; i++) {
        CUDA_CHECK(cudaStreamCreate(&stream[i]));
    }

    CUDA_CHECK(cudaEventRecord(start, 0));

    // initiate all asynchronous transfers to the device
    for (int i = 0; i < NSTREAM; i++) {
        int offset = i * iElem;
        CUDA_CHECK(cudaMemcpyAsync(&d_A[offset], &h_A[offset],
            iBytes, cudaMemcpyHostToDevice, stream[i]));
        CUDA_CHECK(cudaMemcpyAsync(&d_B[offset], &h_B[offset],
            iBytes, cudaMemcpyHostToDevice, stream[i]));
    }

    // launch a kernel in each stream
    for (int i = 0; i < NSTREAM; i++) {
        int offset = i * iElem;
        sumArrays<<<grid, block, 0, stream[i]>>>(&d_A[offset], &d_B[offset], &d_C[offset], iElem);
    }

    // enqueue asynchronous transfers from the device
    for (int i = 0; i < NSTREAM; i++) {
        int offset = i * iElem;
        CUDA_CHECK(cudaMemcpyAsync(&gpuRef[offset], &d_C[offset],
            iBytes, cudaMemcpyDeviceToHost, stream[i]));
    }

    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float execution_time;
    CUDA_CHECK(cudaEventElapsedTime(&execution_time, start, stop));

    printf("\n");
    printf("Actual results from overlapped data transfers:\n");
    printf(" overlap with %d streams : %f ms (%f GB/s)\n", NSTREAM,
           execution_time, (nBytes * 2e-6) / execution_time );
    printf(" speedup                : %f \n",
           ((itotal - execution_time) * 100.0f) / itotal);
    
    // check kernel error
    CUDA_CHECK(cudaGetLastError());

    // check device results
    checkResult(hostRef, gpuRef, nElem);

    // free device global memory
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    // free host memory
    CUDA_CHECK(cudaFreeHost(h_A));
    CUDA_CHECK(cudaFreeHost(h_B));
    CUDA_CHECK(cudaFreeHost(hostRef));
    CUDA_CHECK(cudaFreeHost(gpuRef));

    // destroy events
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // destroy streams
    for (int i = 0; i < NSTREAM; i++) {
        CUDA_CHECK(cudaStreamDestroy(stream[i]));
    }

    CUDA_CHECK(cudaDeviceReset());
    return 0;
}