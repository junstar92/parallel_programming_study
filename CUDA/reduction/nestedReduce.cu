/*****************************************************************************
 * File:        nestedReduce.cu
 * Description: Implement kernel functions for reduce problem(sum)
 *                  - cpuRecursiveReduce
 *                  - reduceNeighbored
 *              
 * Compile:     nvcc -arch=sm_75 -rdc=true -o nestedReduce nestedReduce.cu -I..
 * Run:         ./nestedReduce [N]
 * Argument:    N = block size (1D)
 *****************************************************************************/
#include <stdio.h>
#include <cuda_runtime.h>
#include <common/common.h>

// Recursive Implementation of Interleaved Pair Approach
int cpuRecursiveReduce(int* data, int const size)
{
    if (size == 1)
        return data[0];
    
    int const stride = size / 2;

    for (int i = 0; i < stride; i++)
        data[i] += data[i + stride];
    
    return cpuRecursiveReduce(data, stride);
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

__global__
void gpuRecursiveReduce(int *g_iData, int *g_oData, unsigned int iSize)
{
    unsigned int tid = threadIdx.x;

    int *iData = g_iData + blockIdx.x*blockDim.x;
    int *oData = g_oData + blockIdx.x;

    // stop condition
    if (iSize == 2 && tid == 0) {
        g_oData[blockIdx.x] = iData[0] + iData[1];
        return;
    }

    // nested invocation
    int iStride = iSize >> 1;

    if (iStride > 1 && tid < iStride) {
        // in-place reduction
        iData[tid] += iData[tid + iStride];
    }
    __syncthreads();

    // nested invocation to generate child grids
    if (tid == 0) {
        gpuRecursiveReduce<<<1, iStride>>>(iData, oData, iStride);
        // sync all child grids launched in this block
        cudaDeviceSynchronize();
    }

    __syncthreads(); // sync at block level again
}

__global__
void gpuRecursiveReduceNosync(int *g_iData, int *g_oData, unsigned int iSize)
{
    unsigned int tid = threadIdx.x;

    int *iData = g_iData + blockIdx.x*blockDim.x;
    int *oData = g_oData + blockIdx.x;

    // stop condition
    if (iSize == 2 && tid == 0) {
        g_oData[blockIdx.x] = iData[0] + iData[1];
        return;
    }

    // nested invocation
    int iStride = iSize >> 1;

    if (iStride > 1 && tid < iStride) {
        // in-place reducetion
        iData[tid] += iData[tid + iStride];

        if (tid == 0) {
            gpuRecursiveReduceNosync<<<1, iStride>>>(iData, oData, iStride);
        }
    }
}

__global__
void gpuRecursiveReduce2(int *g_iData, int *g_oData, int iStride, int const iDim)
{
    int *iData = g_iData + blockIdx.x * iDim;

    // stop condition
    if (iStride == 1 && threadIdx.x == 0) {
        g_oData[blockIdx.x] = iData[0] + iData[1];
        return;
    }

    // in-place reduction
    iData[threadIdx.x] += iData[threadIdx.x + iStride];

    // nested invocation to generate child grids
    if (threadIdx.x == 0 && blockIdx.x == 0)
        gpuRecursiveReduce2<<<gridDim.x, iStride / 2>>>(g_iData, g_oData, iStride / 2, iDim);
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

    // execution configuration
    int nBlock = 2048;
    int nThread = 512;

    if (argc > 1)
        nBlock = atoi(argv[1]);
    if (argc > 2)
        nThread = atoi(argv[2]);
    
    int size = nBlock * nThread; // total number of elements

    dim3 block(nThread, 1);
    dim3 grid((size+block.x-1) / block.x, 1);
    printf("array %d grid %d block %d\n", size, grid.x, block.x);

    // allocate host memory
    size_t bytes = size * sizeof(int);
    int* h_iData = (int*)malloc(bytes);
    int* h_oData = (int*)malloc(grid.x * sizeof(int));
    int* tmp = (int*)malloc(bytes);

    // init the array
    for (int i = 0; i < size; i++) {
        h_iData[i] = (int)(rand() & 0xFF);
    }
    
    memcpy(tmp, h_iData, bytes);

    double start, finish;
    int gpu_sum = 0;

    // allocate device memory
    int *d_iData = NULL, *d_oData = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_iData, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_oData, grid.x * sizeof(int)));

    
    // cpu reduction
    GET_TIME(start);
    int cpu_sum = cpuRecursiveReduce(tmp, size);
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
    printf("gpu Neighbored      elapsed %.4f ms gpu_sum: %d <<<grid %d block %d>>>\n",
        (finish-start)*1000.f, gpu_sum, grid.x, block.x);
    
    // kernel 2: gpu nested reduce kernel with synchronization
    CUDA_CHECK(cudaMemcpy(d_iData, h_iData, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
    GET_TIME(start);
    gpuRecursiveReduce<<<grid, block>>>(d_iData, d_oData, block.x);
    CUDA_CHECK(cudaDeviceSynchronize());
    GET_TIME(finish);
    CUDA_CHECK(cudaMemcpy(h_oData, d_oData, grid.x * sizeof(int), cudaMemcpyDeviceToHost));

    gpu_sum = 0;
    for (int i = 0; i < grid.x; i++)
        gpu_sum += h_oData[i];
    printf("gpu Nested          elapsed %.4f ms gpu_sum: %d <<<grid %d block %d>>>\n",
        (finish-start)*1000.f, gpu_sum, grid.x, block.x);
    
    // kernel 3: gpu nested reduce kernel without synchronization
    CUDA_CHECK(cudaMemcpy(d_iData, h_iData, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
    GET_TIME(start);
    gpuRecursiveReduceNosync<<<grid, block>>>(d_iData, d_oData, block.x);
    CUDA_CHECK(cudaDeviceSynchronize());
    GET_TIME(finish);
    CUDA_CHECK(cudaMemcpy(h_oData, d_oData, grid.x * sizeof(int), cudaMemcpyDeviceToHost));

    gpu_sum = 0;
    for (int i = 0; i < grid.x; i++)
        gpu_sum += h_oData[i];
    printf("gpu NestedNosync    elapsed %.4f ms gpu_sum: %d <<<grid %d block %d>>>\n",
        (finish-start)*1000.f, gpu_sum, grid.x, block.x);
    
    // kernel 4: gpu nested reduce kernel 2
    CUDA_CHECK(cudaMemcpy(d_iData, h_iData, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
    GET_TIME(start);
    gpuRecursiveReduce2<<<grid, block.x / 2>>>(d_iData, d_oData, block.x / 2, block.x);
    CUDA_CHECK(cudaDeviceSynchronize());
    GET_TIME(finish);
    CUDA_CHECK(cudaMemcpy(h_oData, d_oData, grid.x * sizeof(int), cudaMemcpyDeviceToHost));

    gpu_sum = 0;
    for (int i = 0; i < grid.x; i++)
        gpu_sum += h_oData[i];
    printf("gpu Nested2         elapsed %.4f ms gpu_sum: %d <<<grid %d block %d>>>\n",
        (finish-start)*1000.f, gpu_sum, grid.x, block.x);

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