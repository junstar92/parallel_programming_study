/*****************************************************************************
 * File:        reduceInteger.cu
 * Description: This is an example of using shared memory to optimize performance
 *              of a parallel reduction by constructing partial results for 
 *              a thread block in shared memory before flushing to global memory.
 *              
 * Compile:     nvcc -o reduceInteger reduceInteger.cu -I..
 * Run:         ./reduceInteger
 *****************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "common/common.h"

#define DIM 128
#define SMEMDIM 4   // 128 / 32 = 4

extern __shared__ int dsmem[];

// Recursive Implementation of Interleaved Pair Approach
int recursiveReduce(int* data, int const size)
{
    if (size == 1)
        return data[0];
    
    int const stride = size / 2;

    for (int i = 0; i < stride; i++)
        data[i] += data[i + stride];
    
    return recursiveReduce(data, stride);
}

// unroll4 + complete unroll for loop + gmem
__global__
void reduceGmem(int* g_iData, int* g_oData, unsigned int n)
{
    // set thread ID
    unsigned int tid = threadIdx.x;

    // boundary check
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n)
        return;
    
    // convert global data pointer to the local pointer of this block
    int* iData = g_iData + (blockDim.x * blockIdx.x);
    
    // in-place reduction in global memory
    if (blockDim.x >= 1024 && tid < 512)
        iData[tid] += iData[tid + 512];
    __syncthreads();

    if (blockDim.x >= 512 && tid < 256)
        iData[tid] += iData[tid + 256];
    __syncthreads();

    if (blockDim.x >= 256 && tid < 128)
        iData[tid] += iData[tid + 128];
    __syncthreads();

    if (blockDim.x >= 128 && tid < 64)
        iData[tid] += iData[tid + 64];
    __syncthreads();

    // unrolling warp
    if (tid < 32) {
        volatile int* vsmem = iData;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid +  8];
        vsmem[tid] += vsmem[tid +  4];
        vsmem[tid] += vsmem[tid +  2];
        vsmem[tid] += vsmem[tid +  1];
    }

    // write result for this block to global memory
    if (tid == 0)
        g_oData[blockIdx.x] = iData[0];
}

__global__
void reduceSmem(int* g_iData, int* g_oData, unsigned int n)
{
    __shared__ int smem[DIM];

    // set thread ID
    unsigned int tid = threadIdx.x;

    // boundary check
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n)
        return;
    
    // convert global data pointer to the local pointer of this block
    int* iData = g_iData + (blockDim.x * blockIdx.x);
    
    // set to smem by each threads
    smem[tid] = iData[tid];
    __syncthreads();

    // in-place reduction in global memory
    if (blockDim.x >= 1024 && tid < 512)
        smem[tid] += smem[tid + 512];
    __syncthreads();

    if (blockDim.x >= 512 && tid < 256)
        smem[tid] += smem[tid + 256];
    __syncthreads();

    if (blockDim.x >= 256 && tid < 128)
        smem[tid] += smem[tid + 128];
    __syncthreads();

    if (blockDim.x >= 128 && tid < 64)
        smem[tid] += smem[tid + 64];
    __syncthreads();

    // unrolling warp
    if (tid < 32) {
        volatile int* vsmem = smem;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid +  8];
        vsmem[tid] += vsmem[tid +  4];
        vsmem[tid] += vsmem[tid +  2];
        vsmem[tid] += vsmem[tid +  1];
    }

    // write result for this block to global memory
    if (tid == 0)
        g_oData[blockIdx.x] = smem[0];
}

__global__
void reduceSmemDyn(int* g_iData, int* g_oData, unsigned int n)
{
    extern __shared__ int smem[];

    // set thread ID
    unsigned int tid = threadIdx.x;

    // boundary check
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n)
        return;
    
    // convert global data pointer to the local pointer of this block
    int* iData = g_iData + (blockDim.x * blockIdx.x);
    
    // set to smem by each threads
    smem[tid] = iData[tid];
    __syncthreads();

    // in-place reduction in global memory
    if (blockDim.x >= 1024 && tid < 512)
        smem[tid] += smem[tid + 512];
    __syncthreads();

    if (blockDim.x >= 512 && tid < 256)
        smem[tid] += smem[tid + 256];
    __syncthreads();

    if (blockDim.x >= 256 && tid < 128)
        smem[tid] += smem[tid + 128];
    __syncthreads();

    if (blockDim.x >= 128 && tid < 64)
        smem[tid] += smem[tid + 64];
    __syncthreads();

    // unrolling warp
    if (tid < 32) {
        volatile int* vsmem = smem;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid +  8];
        vsmem[tid] += vsmem[tid +  4];
        vsmem[tid] += vsmem[tid +  2];
        vsmem[tid] += vsmem[tid +  1];
    }

    // write result for this block to global memory
    if (tid == 0)
        g_oData[blockIdx.x] = smem[0];
}

// unroll4 + complete unroll for loop + gmem
__global__
void reduceGmemUnroll(int* g_iData, int* g_oData, unsigned int n)
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockDim.x * blockIdx.x * 4 + threadIdx.x;
    
    // convert global data pointer to the local pointer of this block
    int* iData = g_iData + (blockDim.x * blockIdx.x * 4);

    // unrolling 4
    if (idx + 3 * blockDim.x < n)
    {
        int a1 = g_iData[idx];
        int a2 = g_iData[idx + blockDim.x];
        int a3 = g_iData[idx + 2 * blockDim.x];
        int a4 = g_iData[idx + 3 * blockDim.x];
        g_iData[idx] = a1 + a2 + a3 + a4;
    }
    __syncthreads();

    // in-place reduction in global memory
    if (blockDim.x >= 1024 && tid < 512)
        iData[tid] += iData[tid + 512];
    __syncthreads();

    if (blockDim.x >= 512 && tid < 256)
        iData[tid] += iData[tid + 256];
    __syncthreads();

    if (blockDim.x >= 256 && tid < 128)
        iData[tid] += iData[tid + 128];
    __syncthreads();

    if (blockDim.x >= 128 && tid < 64)
        iData[tid] += iData[tid + 64];
    __syncthreads();

    // unrolling warp
    if (tid < 32)
    {
        volatile int *vsmem = iData;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid +  8];
        vsmem[tid] += vsmem[tid +  4];
        vsmem[tid] += vsmem[tid +  2];
        vsmem[tid] += vsmem[tid +  1];
    }

    // write result for this block to global mem
    if (tid == 0)
        g_oData[blockIdx.x] = iData[0];
}

__global__
void reduceSmemUnroll(int* g_iData, int* g_oData, unsigned int n)
{
    // static shared memory
    __shared__ int smem[DIM];

    // set thread ID
    unsigned int tid = threadIdx.x;

    // global index, 4 blocks of input data processed at a time
    unsigned int idx = blockDim.x * blockIdx.x * 4 + threadIdx.x;

    // unrolling 4 blocks
    int tmpSum = 0;

    // boundary check
    if (idx + 3 * blockDim.x < n)
    {
        int a1 = g_iData[idx];
        int a2 = g_iData[idx + blockDim.x];
        int a3 = g_iData[idx + 2 * blockDim.x];
        int a4 = g_iData[idx + 3 * blockDim.x];
        tmpSum = a1 + a2 + a3 + a4;
    }
    smem[tid] = tmpSum;
    __syncthreads();

    // in-place reduction in global memory
    if (blockDim.x >= 1024 && tid < 512)
        smem[tid] += smem[tid + 512];
    __syncthreads();

    if (blockDim.x >= 512 && tid < 256)
        smem[tid] += smem[tid + 256];
    __syncthreads();

    if (blockDim.x >= 256 && tid < 128)
        smem[tid] += smem[tid + 128];
    __syncthreads();

    if (blockDim.x >= 128 && tid < 64)
        smem[tid] += smem[tid + 64];
    __syncthreads();

    // unrolling warp
    if (tid < 32) {
        volatile int* vsmem = smem;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid +  8];
        vsmem[tid] += vsmem[tid +  4];
        vsmem[tid] += vsmem[tid +  2];
        vsmem[tid] += vsmem[tid +  1];
    }

    // write result for this block to global memory
    if (tid == 0)
        g_oData[blockIdx.x] = smem[0];
}

__global__
void reduceSmemUnrollDyn(int* g_iData, int* g_oData, unsigned int n)
{
    // static shared memory
    extern __shared__ int smem[];

    // set thread ID
    unsigned int tid = threadIdx.x;

    // global index, 4 blocks of input data processed at a time
    unsigned int idx = blockDim.x * blockIdx.x * 4 + threadIdx.x;

    // unrolling 4 blocks
    int tmpSum = 0;

    // boundary check
    if (idx + 3 * blockDim.x < n)
    {
        int a1 = g_iData[idx];
        int a2 = g_iData[idx + blockDim.x];
        int a3 = g_iData[idx + 2 * blockDim.x];
        int a4 = g_iData[idx + 3 * blockDim.x];
        tmpSum = a1 + a2 + a3 + a4;
    }
    smem[tid] = tmpSum;
    __syncthreads();

    // in-place reduction in global memory
    if (blockDim.x >= 1024 && tid < 512)
        smem[tid] += smem[tid + 512];
    __syncthreads();

    if (blockDim.x >= 512 && tid < 256)
        smem[tid] += smem[tid + 256];
    __syncthreads();

    if (blockDim.x >= 256 && tid < 128)
        smem[tid] += smem[tid + 128];
    __syncthreads();

    if (blockDim.x >= 128 && tid < 64)
        smem[tid] += smem[tid + 64];
    __syncthreads();

    // unrolling warp
    if (tid < 32) {
        volatile int* vsmem = smem;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid +  8];
        vsmem[tid] += vsmem[tid +  4];
        vsmem[tid] += vsmem[tid +  2];
        vsmem[tid] += vsmem[tid +  1];
    }

    // write result for this block to global memory
    if (tid == 0)
        g_oData[blockIdx.x] = smem[0];
}

__inline__ __device__
int warpReduce(int mySum)
{
    mySum += __shfl_xor(mySum, 16);
    mySum += __shfl_xor(mySum, 8);
    mySum += __shfl_xor(mySum, 4);
    mySum += __shfl_xor(mySum, 2);
    mySum += __shfl_xor(mySum, 1);
    return mySum;
}

__global__
void reduceShfl(int* g_iData, int* g_oData, unsigned int n)
{
    // shared memory for each warp sum
    __shared__ int smem[SMEMDIM];

    // boundary check
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n)
        return;
    
    // read from global memory
    int mySum = g_iData[idx];

    // caculate lane index and warp index
    int laneIdx = threadIdx.x % warpSize;
    int warpIdx = threadIdx.x / warpSize;

    // block-wide warp reduce
    mySum = warpReduce(mySum);

    // save warp sum to shared memory
    if (laneIdx == 0)
        smem[warpIdx] = mySum;
    __syncthreads();

    // last warp reduce
    mySum = (threadIdx.x < SMEMDIM) ? smem[laneIdx] : 0;
    if (warpIdx == 0)
        mySum = warpReduce(mySum);
    
    // write reulst for this block to global mem
    if (threadIdx.x == 0)
        g_oData[blockIdx.x] = mySum;
}

int main(int argc, char** argv)
{
    // setup device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Starting reduction at device %d: %s\n", dev, deviceProp.name);

    // initialization
    int size = 1 << 24;
    printf("\twith array size %d  ", size);

    // execution configuration
    dim3 block(DIM, 1);
    dim3 grid((size + block.x - 1) / block.x, 1);
    printf("grid %d block %d\n", grid.x, block.x);

    // allocate host memory
    size_t bytes = size * sizeof(int);
    int* h_iData = (int*)malloc(bytes);
    int* h_oData = (int*)malloc(grid.x * sizeof(int));
    int* tmp = (int*)malloc(bytes);

    // initialize the array
    for (int i = 0; i < size; i++)
        h_iData[i] = (int)(rand() & 0xFF);
    memcpy(tmp, h_iData, bytes);

    int gpu_sum = 0;

    // allocate device memory
    int *d_iData, *d_oData;
    CUDA_CHECK(cudaMalloc((void**)&d_iData, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_oData, grid.x * sizeof(int)));

    // cpu reduction
    int cpu_sum = recursiveReduce(tmp, size);
    printf("cpu reduce          : %d\n", cpu_sum);

    // reduce gmem
    CUDA_CHECK(cudaMemcpy(d_iData, h_iData, bytes, cudaMemcpyHostToDevice));
    reduceGmem<<<grid, block>>>(d_iData, d_oData, size);
    CUDA_CHECK(cudaMemcpy(h_oData, d_oData, grid.x * sizeof(int), cudaMemcpyDeviceToHost));

    gpu_sum = 0;
    for (int i = 0; i < grid.x; i++)
        gpu_sum += h_oData[i];
    printf("reduceGmem          : %d <<<grid %d block %d>>>\n", gpu_sum, grid.x, block.x);

    // reduce smem
    CUDA_CHECK(cudaMemcpy(d_iData, h_iData, bytes, cudaMemcpyHostToDevice));
    reduceSmem<<<grid, block>>>(d_iData, d_oData, size);
    CUDA_CHECK(cudaMemcpy(h_oData, d_oData, grid.x * sizeof(int), cudaMemcpyDeviceToHost));

    gpu_sum = 0;
    for (int i = 0; i < grid.x; i++)
        gpu_sum += h_oData[i];
    printf("reduceSmem          : %d <<<grid %d block %d>>>\n", gpu_sum, grid.x, block.x);

    // reduce smem
    CUDA_CHECK(cudaMemcpy(d_iData, h_iData, bytes, cudaMemcpyHostToDevice));
    reduceSmemDyn<<<grid, block, DIM*sizeof(int)>>>(d_iData, d_oData, size);
    CUDA_CHECK(cudaMemcpy(h_oData, d_oData, grid.x * sizeof(int), cudaMemcpyDeviceToHost));

    gpu_sum = 0;
    for (int i = 0; i < grid.x; i++)
        gpu_sum += h_oData[i];
    printf("reduceSmemDyn       : %d <<<grid %d block %d>>>\n", gpu_sum, grid.x, block.x);

    // reduce gmem
    CUDA_CHECK(cudaMemcpy(d_iData, h_iData, bytes, cudaMemcpyHostToDevice));
    reduceGmemUnroll<<<grid.x / 4, block>>>(d_iData, d_oData, size);
    CUDA_CHECK(cudaMemcpy(h_oData, d_oData, grid.x / 4 * sizeof(int), cudaMemcpyDeviceToHost));

    gpu_sum = 0;
    for (int i = 0; i < grid.x / 4; i++)
        gpu_sum += h_oData[i];
    printf("reduceGmemUnroll4   : %d <<<grid %d block %d>>>\n", gpu_sum, grid.x / 4, block.x);

    // reduce smem
    CUDA_CHECK(cudaMemcpy(d_iData, h_iData, bytes, cudaMemcpyHostToDevice));
    reduceSmemUnroll<<<grid.x / 4, block>>>(d_iData, d_oData, size);
    CUDA_CHECK(cudaMemcpy(h_oData, d_oData, grid.x / 4 * sizeof(int), cudaMemcpyDeviceToHost));

    gpu_sum = 0;
    for (int i = 0; i < grid.x / 4; i++)
        gpu_sum += h_oData[i];
    printf("reduceSmemUnroll4   : %d <<<grid %d block %d>>>\n", gpu_sum, grid.x / 4, block.x);

    // reduce smem
    CUDA_CHECK(cudaMemcpy(d_iData, h_iData, bytes, cudaMemcpyHostToDevice));
    reduceSmemUnrollDyn<<<grid.x / 4, block, DIM*sizeof(int)>>>(d_iData, d_oData, size);
    CUDA_CHECK(cudaMemcpy(h_oData, d_oData, grid.x / 4 * sizeof(int), cudaMemcpyDeviceToHost));

    gpu_sum = 0;
    for (int i = 0; i < grid.x / 4; i++)
        gpu_sum += h_oData[i];
    printf("reduceSmemDynUnroll4: %d <<<grid %d block %d>>>\n", gpu_sum, grid.x / 4, block.x);

    // reduce with warp suffle instrction
    CUDA_CHECK(cudaMemcpy(d_iData, h_iData, bytes, cudaMemcpyHostToDevice));
    reduceShfl<<<grid.x, block, DIM*sizeof(int)>>>(d_iData, d_oData, size);
    CUDA_CHECK(cudaMemcpy(h_oData, d_oData, grid.x * sizeof(int), cudaMemcpyDeviceToHost));

    gpu_sum = 0;
    for (int i = 0; i < grid.x; i++)
        gpu_sum += h_oData[i];
    printf("reduceShfl          : %d <<<grid %d block %d>>>\n", gpu_sum, grid.x, block.x);

    // free host memory
    free(h_iData);
    free(h_oData);

    // free device memory
    CUDA_CHECK(cudaFree(d_iData));
    CUDA_CHECK(cudaFree(d_oData));

    // reset device
    CUDA_CHECK(cudaDeviceReset());

    return 0;
}