/*****************************************************************************
 * File:        matrixAddWithoutUnifiedMemory.cu
 * Description: This is an example to demonstrates the use of explicit CUDA memory
 *              transfer to implement matrix addition. This code contrasts with
 *              matrixAddWithUnifiedMemory.cu, where CUDA managed memory is used to
 *              remove all explicit memory transfers and abstract away the concept
 *              of physicall separate address space.
 *              
 *              
 * Compile:     nvcc -O3 -o manual matrixAddWithoutUnifiedMemory.cu -I..
 * Run:         ./manual
 *                  [n]: power to set size of input matrix (default: 12)
 *****************************************************************************/
#include <stdio.h>
#include <cuda_runtime.h>
#include "common/common.h"

void initialData(float* in, const int size)
{
    for (int i = 0; i < size; i++)
        in[i] = (rand() & 0xFF) / 10.f;
}

void sumMatrixOnHost(float *A, float *B, float *C, const int nx, const int ny)
{
    float* ia = A;
    float* ib = B;
    float* ic = C;

    for (int iy = 0; iy < ny; iy++) {
        for (int ix = 0; ix < nx; ix++) {
            ic[ix] = ia[ix] + ib[ix];
        }

        ia += nx;
        ib += nx;
        ic += nx;
    }
}

void checkResult(float* hostRef, float* gpuRef, const int size)
{
    double epsilon = 1.0e-8;

    for (int i = 0; i < size; i++) {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon) {
            printf("different on %dth element: host %f gpu %f\n", i, hostRef[i], gpuRef[i]);
            break;
        }
    }
}

__global__
void sumMatrixOnGPU(float* A, float* B, float* C, const int nx, const int ny)
{
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int idx = iy * nx + ix;

    if (ix < nx && iy < ny)
        C[idx] = A[idx] + B[idx];
}

int main(int argc, char** argv)
{
    // setup device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Starting sumMatrix at device %d: %s\n", dev, deviceProp.name);
    CUDA_CHECK(cudaSetDevice(dev));

    // setup size of matrix
    int nx, ny;
    int power = 12;
    if (argc > 1)
        power = atoi(argv[1]);
    nx = ny = 1 << power;

    int nxy = nx * ny;
    size_t nBytes = nxy * sizeof(float);
    printf("Matrix size: nx %d ny %d\n", nx, ny);

    // malloc host memory
    float *A, *B, *hostRef, *gpuRef;
    CUDA_CHECK(cudaMallocManaged((void**)&A, nBytes));
    CUDA_CHECK(cudaMallocManaged((void**)&B, nBytes));
    CUDA_CHECK(cudaMallocManaged((void**)&hostRef, nBytes));
    CUDA_CHECK(cudaMallocManaged((void**)&gpuRef, nBytes));

    double start, finish;
    // initialize data at host side
    GET_TIME(start);
    initialData(A, nxy);
    initialData(B, nxy);
    GET_TIME(finish);
    printf("initialization: \t %f sec\n", finish - start);

    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    // add matrix at host side for result check
    GET_TIME(start);
    sumMatrixOnHost(A, B, hostRef, nx, ny);
    GET_TIME(finish);
    printf("sumMatrix on host:\t %f sec\n", finish - start);

    // invode kernel at host side
    int dimX = 32;
    int dimY = 32;
    dim3 blocks(dimX, dimY);
    dim3 grids((nx + blocks.x - 1) / blocks.x, (ny + blocks.y - 1) / blocks.y);
    
    // warm-up kernel
    sumMatrixOnGPU<<<grids, blocks>>>(A, B, gpuRef, nx, ny);
    CUDA_CHECK(cudaDeviceSynchronize());

    GET_TIME(start);
    sumMatrixOnGPU<<<grids, blocks>>>(A, B, gpuRef, nx, ny);
    CUDA_CHECK(cudaDeviceSynchronize());
    GET_TIME(finish);
    printf("sumMatrix on gpu :\t %f sec <<<(%d,%d), (%d,%d)>>>\n", finish-start, grids.x, grids.y, blocks.x, blocks.y);

    // check device results
    checkResult(hostRef, gpuRef, nxy);

    // free device global memory
    CUDA_CHECK(cudaFree(A));
    CUDA_CHECK(cudaFree(B));
    CUDA_CHECK(cudaFree(hostRef));
    CUDA_CHECK(cudaFree(gpuRef));

    CUDA_CHECK(cudaDeviceReset());
    return 0;
}