/*****************************************************************************
 * File:        reduceInteger.cu
 * Description: Implement kernel functions for reduce problem(sum)
 *                  - recursiveReduce
 *                  - reduceNeighbored
 *                  - reduceNeighboredLess
 *                  - reduceInterleaved
 *                  - reduceUnrolling2, reduceUnrolling4, reduceUnrolling8
 *                  - reduceUnrollWarps8
 *                  - reduceCompleteUnrollWarps8
 *                  - reduceCompleteUnroll : template version of reduceCompleteUnrollWarps8
 *              
 * Compile:     nvcc -O3 -arch=sm_75 -o reduceInteger reduceInteger.cu -I..
 * Run:         ./reduceInteger [N]
 * Argument:    N = block size (1D)
 *****************************************************************************/
#include <stdio.h>
#include <cuda_runtime.h>
#include <common/common.h>

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

// Neighbored Pair Implementation with divergence
__global__
void reduceNeighbored(int *g_iData, int *g_oData, unsigned int n)
{
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockDim.x*blockIdx.x + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int *iData = g_iData + (blockIdx.x * blockDim.x);

    // bound check
    if (idx >= n)
        return;
    
    // in-place reduction in global memory
    for (int stride = 1; stride < blockDim.x; stride *= 2)
    {
        if ((tid % (2*stride)) == 0) {
            iData[tid] += iData[tid + stride];
        }

        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0)
        g_oData[blockIdx.x] = iData[0];
}

// Neighbored Pair Implementation with less divergence
__global__
void reduceNeighboredLess(int *g_iData, int *g_oData, unsigned int n)
{
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockDim.x*blockIdx.x + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int *iData = g_iData + (blockIdx.x * blockDim.x);

    // bound check
    if (idx >= n)
        return;
    
    // in-place reduction in global memory
    for (int stride = 1; stride < blockDim.x; stride *= 2)
    {
        // convert tid into local array index
        int index = 2 * stride * tid;

        if (index < blockDim.x)
            iData[index] += iData[index + stride];

        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0)
        g_oData[blockIdx.x] = iData[0];
}

// Interleaved Pair Implementation with less divergence
__global__
void reduceInterleaved(int *g_iData, int *g_oData, unsigned int n)
{
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockDim.x*blockIdx.x + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int *iData = g_iData + (blockIdx.x * blockDim.x);

    // bound check
    if (idx >= n)
        return;

    // in-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
            iData[tid] += iData[tid + stride];

        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0)
        g_oData[blockIdx.x] = iData[0];
}

// unrolling 2
__global__
void reduceUnrolling2(int *g_iData, int *g_oData, unsigned int n)
{
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockDim.x*blockIdx.x*2 + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int *iData = g_iData + (blockIdx.x * blockDim.x * 2);

    // unrolling 2
    if (idx + blockDim.x < n)
        g_iData[idx] += g_iData[idx + blockDim.x];
    
    __syncthreads();

    // in-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
            iData[tid] += iData[tid + stride];

        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0)
        g_oData[blockIdx.x] = iData[0];
}

// unrolling 4
__global__
void reduceUnrolling4(int *g_iData, int *g_oData, unsigned int n)
{
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockDim.x*blockIdx.x*4 + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int *iData = g_iData + (blockIdx.x * blockDim.x * 4);

    // unrolling 4
    if (idx + 3*blockDim.x < n) {
        int a1 = g_iData[idx];
        int a2 = g_iData[idx + blockDim.x];
        int a3 = g_iData[idx + 2*blockDim.x];
        int a4 = g_iData[idx + 3*blockDim.x];
        g_iData[idx] = a1 + a2 + a3 + a4;
    }
    
    __syncthreads();

    // in-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
            iData[tid] += iData[tid + stride];

        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0)
        g_oData[blockIdx.x] = iData[0];
}

// unrolling 8
__global__
void reduceUnrolling8(int *g_iData, int *g_oData, unsigned int n)
{
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockDim.x*blockIdx.x*8 + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int *iData = g_iData + (blockIdx.x * blockDim.x * 8);

    // unrolling 8
    if (idx + 7*blockDim.x < n) {
        int a1 = g_iData[idx];
        int a2 = g_iData[idx + blockDim.x];
        int a3 = g_iData[idx + 2*blockDim.x];
        int a4 = g_iData[idx + 3*blockDim.x];
        int b1 = g_iData[idx + 4*blockDim.x];
        int b2 = g_iData[idx + 5*blockDim.x];
        int b3 = g_iData[idx + 6*blockDim.x];
        int b4 = g_iData[idx + 7*blockDim.x];
        g_iData[idx] = a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4;
    }
    
    __syncthreads();

    // in-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
            iData[tid] += iData[tid + stride];

        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0)
        g_oData[blockIdx.x] = iData[0];
}

// unrolling warps 8
__global__
void reduceUnrollWarps8(int *g_iData, int *g_oData, unsigned int n)
{
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockDim.x*blockIdx.x*8 + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int *iData = g_iData + (blockIdx.x * blockDim.x * 8);

    // unrolling 8
    if (idx + 7*blockDim.x < n) {
        int a1 = g_iData[idx];
        int a2 = g_iData[idx + blockDim.x];
        int a3 = g_iData[idx + 2*blockDim.x];
        int a4 = g_iData[idx + 3*blockDim.x];
        int b1 = g_iData[idx + 4*blockDim.x];
        int b2 = g_iData[idx + 5*blockDim.x];
        int b3 = g_iData[idx + 6*blockDim.x];
        int b4 = g_iData[idx + 7*blockDim.x];
        g_iData[idx] = a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4;
    }
    
    __syncthreads();

    // in-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1)
    {
        if (tid < stride)
            iData[tid] += iData[tid + stride];

        __syncthreads();
    }

    // unrolling warp
    if (tid < 32) {
        volatile int *vmem = iData;
        vmem[tid] += vmem[tid + 32];
        vmem[tid] += vmem[tid + 16];
        vmem[tid] += vmem[tid + 8];
        vmem[tid] += vmem[tid + 4];
        vmem[tid] += vmem[tid + 2];
        vmem[tid] += vmem[tid + 1];
    }

    // write result for this block to global mem
    if (tid == 0)
        g_oData[blockIdx.x] = iData[0];
}

// complete unroll warp
__global__
void reduceCompleteUnrollWarps8(int *g_iData, int *g_oData, unsigned int n)
{
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockDim.x*blockIdx.x*8 + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int *iData = g_iData + (blockIdx.x * blockDim.x * 8);

    // unrolling 8
    if (idx + 7*blockDim.x < n) {
        int a1 = g_iData[idx];
        int a2 = g_iData[idx + blockDim.x];
        int a3 = g_iData[idx + 2*blockDim.x];
        int a4 = g_iData[idx + 3*blockDim.x];
        int b1 = g_iData[idx + 4*blockDim.x];
        int b2 = g_iData[idx + 5*blockDim.x];
        int b3 = g_iData[idx + 6*blockDim.x];
        int b4 = g_iData[idx + 7*blockDim.x];
        g_iData[idx] = a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4;
    }
    
    __syncthreads();

    // in-place reduction and complete unroll
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
        volatile int *vmem = iData;
        vmem[tid] += vmem[tid + 32];
        vmem[tid] += vmem[tid + 16];
        vmem[tid] += vmem[tid + 8];
        vmem[tid] += vmem[tid + 4];
        vmem[tid] += vmem[tid + 2];
        vmem[tid] += vmem[tid + 1];
    }

    // write result for this block to global mem
    if (tid == 0)
        g_oData[blockIdx.x] = iData[0];
}

template<unsigned int iBlockSize>
__global__
void reduceCompleteUnroll(int *g_iData, int *g_oData, unsigned int n)
{
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockDim.x*blockIdx.x*8 + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int *iData = g_iData + (blockIdx.x * blockDim.x * 8);

    // unrolling 8
    if (idx + 7*blockDim.x < n) {
        int a1 = g_iData[idx];
        int a2 = g_iData[idx + blockDim.x];
        int a3 = g_iData[idx + 2*blockDim.x];
        int a4 = g_iData[idx + 3*blockDim.x];
        int b1 = g_iData[idx + 4*blockDim.x];
        int b2 = g_iData[idx + 5*blockDim.x];
        int b3 = g_iData[idx + 6*blockDim.x];
        int b4 = g_iData[idx + 7*blockDim.x];
        g_iData[idx] = a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4;
    }
    
    __syncthreads();

    // in-place reduction and complete unroll
    if (iBlockSize >= 1024 && tid < 512)
        iData[tid] += iData[tid + 512];
    __syncthreads();

    if (iBlockSize >= 512 && tid < 256)
        iData[tid] += iData[tid + 256];
    __syncthreads();

    if (iBlockSize >= 256 && tid < 128)
        iData[tid] += iData[tid + 128];
    __syncthreads();

    if (iBlockSize >= 128 && tid < 64)
        iData[tid] += iData[tid + 64];
    __syncthreads();

    // unrolling warp
    if (tid < 32) {
        volatile int *vmem = iData;
        vmem[tid] += vmem[tid + 32];
        vmem[tid] += vmem[tid + 16];
        vmem[tid] += vmem[tid + 8];
        vmem[tid] += vmem[tid + 4];
        vmem[tid] += vmem[tid + 2];
        vmem[tid] += vmem[tid + 1];
    }

    // write result for this block to global mem
    if (tid == 0)
        g_oData[blockIdx.x] = iData[0];
}

int main(int argc, char** argv)
{
    // setup device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Starting reduction at ");
    printf("device %d: %s ", dev, deviceProp.name);
    CUDA_CHECK(cudaSetDevice(dev));

    bool result = false;

    // init
    int size = 1 << 24;
    printf("    with array size %d ", size);

    // execution configuration
    int blockSize = 512;

    if (argc > 1)
        blockSize = atoi(argv[1]);
    
    dim3 block(blockSize, 1);
    dim3 grid((size+block.x-1) / block.x, 1);
    printf("grid %d block %d\n", grid.x, block.x);

    // allocate host memory
    size_t bytes = size * sizeof(int);
    int* h_iData = (int*)malloc(bytes);
    int* h_oData = (int*)malloc(grid.x * sizeof(int));
    int* tmp = (int*)malloc(bytes);

    // init the array
    for (int i = 0; i < size; i++)
        h_iData[i] = (int)(rand() & 0xFF);
    
    memcpy(tmp, h_iData, bytes);

    double start, finish;
    int gpu_sum = 0;

    // allocate device memory
    int *d_iData = NULL, *d_oData = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_iData, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_oData, grid.x * sizeof(int)));

    
    // cpu reduction
    GET_TIME(start);
    int cpu_sum = recursiveReduce(tmp, size);
    GET_TIME(finish);
    printf("cpu reduce          elapsed %.4f ms cpu_sum: %d\n", (finish-start)*1000.f, cpu_sum);

    // kernel 1: reduceNeighbored
    CUDA_CHECK(cudaMemcpy(d_iData, h_iData, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
    GET_TIME(start);
    reduceNeighbored<<<grid, block>>>(d_iData, d_oData, size);
    CUDA_CHECK(cudaDeviceSynchronize());
    GET_TIME(finish);
    CUDA_CHECK(cudaMemcpy(h_oData, d_oData, grid.x * sizeof(int), cudaMemcpyDeviceToHost));

    gpu_sum = 0;
    for (int i = 0; i < grid.x; i++)
        gpu_sum += h_oData[i];
    printf("gpu Neighbored      elapsed %.4f ms gpu_sum: %d <<<grud %d block %d>>>\n",
        (finish-start)*1000.f, gpu_sum, grid.x, block.x);
    
    // kernel 2: reduceNeighbored with less divergence
    CUDA_CHECK(cudaMemcpy(d_iData, h_iData, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
    GET_TIME(start);
    reduceNeighboredLess<<<grid, block>>>(d_iData, d_oData, size);
    CUDA_CHECK(cudaDeviceSynchronize());
    GET_TIME(finish);
    CUDA_CHECK(cudaMemcpy(h_oData, d_oData, grid.x * sizeof(int), cudaMemcpyDeviceToHost));

    gpu_sum = 0;
    for (int i = 0; i < grid.x; i++)
        gpu_sum += h_oData[i];
    printf("gpu Neighbored2     elapsed %.4f ms gpu_sum: %d <<<grud %d block %d>>>\n",
        (finish-start)*1000.f, gpu_sum, grid.x, block.x);

    // kernel 3: reduceInterleaved
    CUDA_CHECK(cudaMemcpy(d_iData, h_iData, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
    GET_TIME(start);
    reduceInterleaved<<<grid, block>>>(d_iData, d_oData, size);
    CUDA_CHECK(cudaDeviceSynchronize());
    GET_TIME(finish);
    CUDA_CHECK(cudaMemcpy(h_oData, d_oData, grid.x * sizeof(int), cudaMemcpyDeviceToHost));

    gpu_sum = 0;
    for (int i = 0; i < grid.x; i++)
        gpu_sum += h_oData[i];
    printf("gpu Interleaved     elapsed %.4f ms gpu_sum: %d <<<grud %d block %d>>>\n",
        (finish-start)*1000.f, gpu_sum, grid.x, block.x);

    // kernel 4: reduceUnrolling2
    CUDA_CHECK(cudaMemcpy(d_iData, h_iData, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
    GET_TIME(start);
    reduceUnrolling2<<<grid.x / 2, block>>>(d_iData, d_oData, size);
    CUDA_CHECK(cudaDeviceSynchronize());
    GET_TIME(finish);
    CUDA_CHECK(cudaMemcpy(h_oData, d_oData, grid.x / 2 * sizeof(int), cudaMemcpyDeviceToHost));

    gpu_sum = 0;
    for (int i = 0; i < grid.x / 2; i++)
        gpu_sum += h_oData[i];
    printf("gpu Unrolling2      elapsed %.4f ms gpu_sum: %d <<<grud %d block %d>>>\n",
        (finish-start)*1000.f, gpu_sum, grid.x/2, block.x);

    // kernel 5: reduceUnrolling4
    CUDA_CHECK(cudaMemcpy(d_iData, h_iData, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
    GET_TIME(start);
    reduceUnrolling4<<<grid.x / 4, block>>>(d_iData, d_oData, size);
    CUDA_CHECK(cudaDeviceSynchronize());
    GET_TIME(finish);
    CUDA_CHECK(cudaMemcpy(h_oData, d_oData, grid.x / 4 * sizeof(int), cudaMemcpyDeviceToHost));

    gpu_sum = 0;
    for (int i = 0; i < grid.x / 4; i++)
        gpu_sum += h_oData[i];
    printf("gpu Unrolling4      elapsed %.4f ms gpu_sum: %d <<<grud %d block %d>>>\n",
        (finish-start)*1000.f, gpu_sum, grid.x/4, block.x);

    // kernel 6: reduceUnrolling8
    CUDA_CHECK(cudaMemcpy(d_iData, h_iData, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
    GET_TIME(start);
    reduceUnrolling8<<<grid.x / 8, block>>>(d_iData, d_oData, size);
    CUDA_CHECK(cudaDeviceSynchronize());
    GET_TIME(finish);
    CUDA_CHECK(cudaMemcpy(h_oData, d_oData, grid.x / 8 * sizeof(int), cudaMemcpyDeviceToHost));

    gpu_sum = 0;
    for (int i = 0; i < grid.x / 8; i++)
        gpu_sum += h_oData[i];
    printf("gpu Unrolling8      elapsed %.4f ms gpu_sum: %d <<<grud %d block %d>>>\n",
        (finish-start)*1000.f, gpu_sum, grid.x/8, block.x);
        
    // kernel 7: reduceUnrollWarps8
    CUDA_CHECK(cudaMemcpy(d_iData, h_iData, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
    GET_TIME(start);
    reduceUnrollWarps8<<<grid.x / 8, block>>>(d_iData, d_oData, size);
    CUDA_CHECK(cudaDeviceSynchronize());
    GET_TIME(finish);
    CUDA_CHECK(cudaMemcpy(h_oData, d_oData, grid.x / 8 * sizeof(int), cudaMemcpyDeviceToHost));

    gpu_sum = 0;
    for (int i = 0; i < grid.x / 8; i++)
        gpu_sum += h_oData[i];
    printf("gpu UnrollWarp8     elapsed %.4f ms gpu_sum: %d <<<grud %d block %d>>>\n",
        (finish-start)*1000.f, gpu_sum, grid.x/8, block.x);

    // kernel 8: reduceCompleteUnrollWarps8
    CUDA_CHECK(cudaMemcpy(d_iData, h_iData, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
    GET_TIME(start);
    reduceCompleteUnrollWarps8<<<grid.x / 8, block>>>(d_iData, d_oData, size);
    CUDA_CHECK(cudaDeviceSynchronize());
    GET_TIME(finish);
    CUDA_CHECK(cudaMemcpy(h_oData, d_oData, grid.x / 8 * sizeof(int), cudaMemcpyDeviceToHost));

    gpu_sum = 0;
    for (int i = 0; i < grid.x / 8; i++)
        gpu_sum += h_oData[i];
    printf("gpu CompleteUnroll8 elapsed %.4f ms gpu_sum: %d <<<grud %d block %d>>>\n",
        (finish-start)*1000.f, gpu_sum, grid.x/8, block.x);

    // kernel 9: reduceCompleteUnroll
    CUDA_CHECK(cudaMemcpy(d_iData, h_iData, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
    GET_TIME(start);
    switch (blockSize) {
    case 1024:
        reduceCompleteUnroll<1024><<<grid.x / 8, block>>>(d_iData, d_oData, size);
        break;
    case 512:
        reduceCompleteUnroll<512><<<grid.x / 8, block>>>(d_iData, d_oData, size);
        break;
    case 256:
        reduceCompleteUnroll<256><<<grid.x / 8, block>>>(d_iData, d_oData, size);
        break;
    case 128:
        reduceCompleteUnroll<128><<<grid.x / 8, block>>>(d_iData, d_oData, size);
        break;
    case 64:
        reduceCompleteUnroll<64><<<grid.x / 8, block>>>(d_iData, d_oData, size);
        break;
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    GET_TIME(finish);
    CUDA_CHECK(cudaMemcpy(h_oData, d_oData, grid.x / 8 * sizeof(int), cudaMemcpyDeviceToHost));

    gpu_sum = 0;
    for (int i = 0; i < grid.x / 8; i++)
        gpu_sum += h_oData[i];
    printf("gpu CompleteUnroll  elapsed %.4f ms gpu_sum: %d <<<grud %d block %d>>>\n",
        (finish-start)*1000.f, gpu_sum, grid.x/8, block.x);

    // free host memory
    free(h_iData);
    free(h_oData);

    // free device memory
    CUDA_CHECK(cudaFree(d_iData));
    CUDA_CHECK(cudaFree(d_oData));

    // reset device
    CUDA_CHECK(cudaDeviceReset());

    // check the results
    result = (gpu_sum == cpu_sum);
    if (!result)
        printf("Test failed!\n");
    
    return 0;
}