/*****************************************************************************
 * File:        constantStencil.cu
 * Description: This is an example of using constant memory to optimize performance
 *              of a stencil computation by storing coefficients of the computation
 *              in a constant memory array (coef).
 *              
 * Compile:     nvcc -o constantStencil constantStencil.cu -I..
 * Run:         ./constantStencil
 *****************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "common/common.h"

#define RADIUS 4
#define BDIM 32

// constant memory
__constant__ float coef[RADIUS + 1];

// FD coeffecient
#define a0     0.00000f
#define a1     0.80000f
#define a2    -0.20000f
#define a3     0.03809f
#define a4    -0.00357f

void initialData(float* in, const int size)
{
    for (int i = 0; i < size; i++)
        in[i] = (rand() & 0xFF) / 100.f;
}

void setup_coef_constant()
{
    const float h_coef[] = {a0, a1, a2, a3, a4};
    CUDA_CHECK(cudaMemcpyToSymbol(coef, h_coef, (RADIUS + 1) * sizeof(float)));
}

void cpu_stencil_1d(float* in, float* out, const int size)
{
    for (int i = RADIUS; i <= size; i++) {
        float tmp = a1 * (in[i + 1] - in[i - 1])
                    + a2 * (in[i + 2] - in[i - 2])
                    + a3 * (in[i + 3] - in[i - 3])
                    + a4 * (in[i + 4] - in[i - 4]);
        out[i] = tmp;
    }
}

void checkResult(float* hostRef, float* gpuRef, const int size)
{
    double epsilon = 1.e-6;

    for (int i = RADIUS; i < size; i++) {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon) {
            printf("different on %dth element: host %f gpu %f\n", i, hostRef[i], gpuRef[i]);
            printf("Array do not match.\n\n");
            break;
        }
    }
}

__global__
void stencil_1d(float* in, float* out, const int N)
{
    // shared memory
    __shared__ float smem[BDIM + 2 * RADIUS];

    // index to global memory
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    // index to shared memory for stencil caculation
    int sidx = threadIdx.x + RADIUS;

    // Read data from global memory into shared memory
    smem[sidx] = in[idx];

    // read halo part to shared memory
    if (threadIdx.x < RADIUS) {
        smem[sidx - RADIUS] = in[idx - RADIUS];
        smem[sidx + BDIM] = in[idx + BDIM];
    }

    // sync
    __syncthreads();

    // apply the stencil
    float tmp = 0.f;

#pragma unroll
    for (int i = 1; i <= RADIUS; i++)
        tmp += coef[i] * (smem[sidx + i] - smem[sidx - i]);
    
    // store the result
    out[idx] = tmp;
}

__global__
void stencil_1d_read_only(float* in, float* out, const float *__restrict__ dcoef)
{
    // shared memory
    __shared__ float smem[BDIM + 2 * RADIUS];

    // index to global memory
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    // index to shared memory for stencil caculation
    int sidx = threadIdx.x + RADIUS;

    // Read data from global memory into shared memory
    smem[sidx] = in[idx];

    // read halo part to shared memory
    if (threadIdx.x < RADIUS) {
        smem[sidx - RADIUS] = in[idx - RADIUS];
        smem[sidx + BDIM] = in[idx + BDIM];
    }

    // sync
    __syncthreads();

    // apply the stencil
    float tmp = 0.f;
    
#pragma unroll
    for (int i = 1; i <= RADIUS; i++)
        tmp += dcoef[i] * (smem[sidx + i] - smem[sidx - i]);
    
    // store the result
    out[idx] = tmp;
}

int main(int argc, char** argv)
{
    // setup device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Starting transpose at device %d: %s\n", dev, deviceProp.name);
    CUDA_CHECK(cudaSetDevice(dev));

    // setup data size
    int size = 32;
    size_t nBytes = (size + 2 * RADIUS) * sizeof(float);
    printf("  array size: %d ", size);

    // allocate host memory
    float* h_in = (float*)malloc(nBytes);
    float* hostRef = (float*)malloc(nBytes);
    float* gpuRef = (float*)malloc(nBytes);

    // allocate device memory
    float* d_in, *d_out, *d_coef;
    CUDA_CHECK(cudaMalloc((float**)&d_in, nBytes));
    CUDA_CHECK(cudaMalloc((float**)&d_out, nBytes));
    CUDA_CHECK(cudaMalloc((void**)&d_coef, (RADIUS+1) * sizeof(float)));

    // setup coefficient to global memory
    const float h_coef[] = {a0, a1, a2, a3, a4};
    CUDA_CHECK(cudaMemcpy(d_coef, h_coef, (RADIUS + 1) * sizeof(float), cudaMemcpyHostToDevice));

    // initialize host array
    initialData(h_in, size + 2 * RADIUS);

    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_in, h_in, nBytes, cudaMemcpyHostToDevice));

    // setup constant memory
    setup_coef_constant();

    // launch configuration
    dim3 block(BDIM, 1);
    dim3 grid(deviceProp.maxGridSize[0] < size / block.x ? deviceProp.maxGridSize[0] : size / block.x, 1);
    printf("(grid, block) %d,%d \n", grid.x, block.x);

    // launch kernel on GPU
    stencil_1d<<<grid, block>>>(d_in + RADIUS, d_out + RADIUS, size);

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(gpuRef, d_out, nBytes, cudaMemcpyDeviceToHost));

    // apply cpu stencil
    cpu_stencil_1d(h_in, hostRef, size);

    // check results
    checkResult(hostRef, gpuRef, size);

    // launch read only cache kernel
    stencil_1d_read_only<<<grid, block>>>(d_in + RADIUS, d_out + RADIUS, d_coef);
    CUDA_CHECK(cudaMemcpy(gpuRef, d_out, nBytes, cudaMemcpyDeviceToHost));
    checkResult(hostRef, gpuRef, size);

    // free memory
    free(h_in);
    free(hostRef);
    free(gpuRef);
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));

    // reset device
    CUDA_CHECK(cudaDeviceReset());
    return 0;
}