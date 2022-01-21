/*****************************************************************************
 * File:        simpleShuffle.cu
 * Description: This is an example of a varienty of shuffle instructions
 *              
 * Compile:     nvcc -o simpleShuffle simpleShuffle.cu -I..
 * Run:         ./simpleShuffle
 *****************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "common/common.h"

#define BDIMX 16
#define SEGM 4

void printData(int* in, const int size)
{
    for (int i = 0; i < size; i++)
        printf("%2d ", in[i]);
    printf("\n");
}

__global__
void test_shfl_broadcast(int* in, int* out, const int srcLane)
{
    int value = in[threadIdx.x];
    value = __shfl(value, srcLane, BDIMX);
    out[threadIdx.x] = value;
}

__global__
void test_shfl_up(int* in, int* out, const unsigned int delta)
{
    int value = in[threadIdx.x];
    value = __shfl_up(value, delta, BDIMX);
    out[threadIdx.x] = value;
}

__global__
void test_shfl_down(int* in, int* out, const unsigned int delta)
{
    int value = in[threadIdx.x];
    value = __shfl_down(value, delta, BDIMX);
    out[threadIdx.x] = value;
}

__global__
void test_shfl_warp(int* in, int* out, const int offset)
{
    int value = in[threadIdx.x];
    value = __shfl(value, threadIdx.x + offset, BDIMX);
    out[threadIdx.x] = value;
}

__global__
void test_shfl_xor(int* in, int* out, const int mask)
{
    int value = in[threadIdx.x];
    value = __shfl_xor(value, mask, BDIMX);
    out[threadIdx.x] = value;
}

__global__
void test_shfl_xor_array(int* in, int* out, const int mask)
{
    int idx = threadIdx.x * SEGM;
    int value[SEGM];

    for (int i = 0; i < SEGM; i++)
        value[i] = in[idx + i];
    
    value[0] = __shfl_xor(value[0], mask, BDIMX);
    value[1] = __shfl_xor(value[1], mask, BDIMX);
    value[2] = __shfl_xor(value[2], mask, BDIMX);
    value[3] = __shfl_xor(value[3], mask, BDIMX);

    for (int i = 0; i < SEGM; i++)
        out[idx + i] = value[i];
}

__inline__ __device__
void swap(int *value, int laneIdx, int mask, int firstIdx, int secondIdx)
{
    bool pred = ((laneIdx / mask + 1) == 1);
    if (pred) {
        int tmp = value[firstIdx];
        value[firstIdx] = value[secondIdx];
        value[secondIdx] = tmp;
    }

    value[secondIdx] = __shfl_xor(value[secondIdx], mask, BDIMX);
    
    if (pred) {
        int tmp = value[firstIdx];
        value[firstIdx] = value[secondIdx];
        value[secondIdx] = tmp;
    }
}

__global__
void test_shfl_swap(int* in, int* out, const int mask, int firstIdx, int secondIdx)
{
    int idx = threadIdx.x * SEGM;
    int value[SEGM];

    for (int i = 0; i < SEGM; i++)
        value[i] = in[idx + i];
    
    swap(value, threadIdx.x, mask, firstIdx, secondIdx);
    
    for (int i = 0; i < SEGM; i++)
        out[idx + i] = value[i];
}

int main(int argc, char** argv)
{
    // setup device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Starting at device %d: %s\n", dev, deviceProp.name);
    CUDA_CHECK(cudaSetDevice(dev));

    int nElem = BDIMX;
    int h_in[BDIMX], h_out[BDIMX];

    for (int i = 0; i < nElem; i++)
        h_in[i] = i;

    size_t nBytes = nElem * sizeof(int);
    int *d_in, *d_out;
    CUDA_CHECK(cudaMalloc((void**)&d_in, nBytes));
    CUDA_CHECK(cudaMalloc((void**)&d_out, nBytes));
    
    CUDA_CHECK(cudaMemcpy(d_in, h_in, nBytes, cudaMemcpyHostToDevice));

    int block = BDIMX;
    /* start test */
    // shuffle broadcast
    test_shfl_broadcast<<<1,block>>>(d_in, d_out, 2);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_out, d_out, nBytes, cudaMemcpyDeviceToHost));

    printf("initialData           : ");
    printData(h_in, nElem);
    printf("shuffle broadcast     : ");
    printData(h_out, nElem);
    printf("\n");

    // shuffle up
    test_shfl_up<<<1,block>>>(d_in, d_out, 2);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_out, d_out, nBytes, cudaMemcpyDeviceToHost));

    printf("initialData           : ");
    printData(h_in, nElem);
    printf("shuffle up            : ");
    printData(h_out, nElem);
    printf("\n");

    // shuffle down
    test_shfl_down<<<1,block>>>(d_in, d_out, 2);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_out, d_out, nBytes, cudaMemcpyDeviceToHost));

    printf("initialData           : ");
    printData(h_in, nElem);
    printf("shuffle down          : ");
    printData(h_out, nElem);
    printf("\n");

    // shuffle offset
    test_shfl_warp<<<1,block>>>(d_in, d_out, 2);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_out, d_out, nBytes, cudaMemcpyDeviceToHost));

    printf("initialData           : ");
    printData(h_in, nElem);
    printf("shuffle offset        : ");
    printData(h_out, nElem);
    printf("\n");

    // shuffle xor
    test_shfl_xor<<<1,block>>>(d_in, d_out, 1);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_out, d_out, nBytes, cudaMemcpyDeviceToHost));

    printf("initialData           : ");
    printData(h_in, nElem);
    printf("shuffle xor           : ");
    printData(h_out, nElem);
    printf("\n");

    // shuffle xor array
    test_shfl_xor_array<<<1,BDIMX / SEGM>>>(d_in, d_out, 1);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_out, d_out, nBytes, cudaMemcpyDeviceToHost));

    printf("initialData           : ");
    printData(h_in, nElem);
    printf("shuffle xor array     : ");
    printData(h_out, nElem);
    printf("\n");

    // shuffle xor swap
    test_shfl_swap<<<1,BDIMX / SEGM>>>(d_in, d_out, 1, 0, 3);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_out, d_out, nBytes, cudaMemcpyDeviceToHost));

    printf("initialData           : ");
    printData(h_in, nElem);
    printf("shuffle swap          : ");
    printData(h_out, nElem);
    printf("\n");

    // free memory
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaDeviceReset());

    return 0;
}