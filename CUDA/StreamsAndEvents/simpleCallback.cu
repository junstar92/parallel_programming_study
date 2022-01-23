/*****************************************************************************
 * File:        simpleCallback.cu
 * Description: This is an example of using CUDA callbacks to trigger work on
 *              the host after the completion of asynchronous work on the device.
 *              In this example, NSTREAM CUDA streams are created and 4 kernels
 *              are launched asynchronously in each. Then, a callback is added
 *              at the completion of those asynchronous kernels that prints
 *              prints diagnostic information.
 *              
 * Compile:     nvcc -o simpleCallback simpleCallback.cu -I..
 * Run:         ./simpleCallback
 *****************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "common/common.h"

#define N 100000
#define NSTREAM 4

void CUDART_CB my_callback(cudaStream_t stream, cudaError_t status, void* data)
{
    printf("callback from stream %d\n", *((int*)data));
}

__global__ void kernel_1()
{
    double sum = 0.0;
    for (int i = 0; i < N; i++) {
        sum = sum + tan(0.1) * tan(0.1);
    }
}

__global__ void kernel_2()
{
    double sum = 0.0;
    for (int i = 0; i < N; i++) {
        sum = sum + tan(0.1) * tan(0.1);
    }
}

__global__ void kernel_3()
{
    double sum = 0.0;
    for (int i = 0; i < N; i++) {
        sum = sum + tan(0.1) * tan(0.1);
    }
}

__global__ void kernel_4()
{
    double sum = 0.0;
    for (int i = 0; i < N; i++) {
        sum = sum + tan(0.1) * tan(0.1);
    }
}

int main(int argc, char** argv)
{    
    int dev = 0;
    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("> Using device %d: %s\n", dev, deviceProp.name);
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
    _putenv_s(iname, "8");
    char* ivalue = getenv(iname);
    printf("> %s = %s\n", iname, ivalue);
    printf("> with streams = %d\n", NSTREAM);

    // allocate and initialize an array of stream handles
    cudaStream_t *streams = (cudaStream_t*)malloc(NSTREAM * sizeof(cudaStream_t));
    for (int i = 0; i < NSTREAM; i++) {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
    }

    dim3 block(1);
    dim3 grid(1);
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    int stream_ids[NSTREAM];

    CUDA_CHECK(cudaEventRecord(start, 0));
    for (int i = 0; i < NSTREAM; i++) {
        stream_ids[i] = i;
        kernel_1<<<grid, block, 0, streams[i]>>>();
        kernel_2<<<grid, block, 0, streams[i]>>>();
        kernel_3<<<grid, block, 0, streams[i]>>>();
        kernel_4<<<grid, block, 0, streams[i]>>>();
        CUDA_CHECK(cudaStreamAddCallback(streams[i], my_callback, (void*)(stream_ids + i), 0));
    }
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_time;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
    printf("Measured time for parallel execution = %.3fs\n", elapsed_time / 1000.f);

    // release all stream
    for (int i = 0; i < NSTREAM; i++) {
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
    }
    free(streams);

    CUDA_CHECK(cudaDeviceReset());
    return 0;
}