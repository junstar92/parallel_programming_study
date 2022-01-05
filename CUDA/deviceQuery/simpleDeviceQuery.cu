/*****************************************************************************
 * File:        simpleDeviceQuery.cu
 * Description: Query device information
 *              
 * Compile:     nvcc -o simpleDeviceQuery simpleDeviceQuery.cu
 * Run:         ./simpleDeviceQuery
 *****************************************************************************/
#include <stdio.h>
#include <cuda_runtime.h>

int main(int argc, char** argv)
{
    int dev = 0;
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, dev);

    printf("Device %d: %s\n", dev, devProp.name);
    printf("Number of multiprocessors: %d\n", devProp.multiProcessorCount);
    printf("Total amount of constant memory: %4.2f KB\n", devProp.totalConstMem/1024.0);
    printf("Total amount of shared memory per block: %4.2f KB\n", devProp.sharedMemPerBlock/1024.0);
    printf("Total number of registers available per block: %d\n", devProp.regsPerBlock);
    printf("Warp size: %d\n", devProp.warpSize);
    printf("Maximum number of threads per multiprocessor: %d\n", devProp.maxThreadsPerMultiProcessor);
    printf("Maximum number of warps per multiprocessor: %d\n", devProp.maxThreadsPerMultiProcessor/devProp.warpSize);

    return 0;
}