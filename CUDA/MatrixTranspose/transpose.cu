/*****************************************************************************
 * File:        transpose.cu
 * Description: This is an example of matrix transpose kernel using various
 *              memory access pattern
 * 
 *              Kernel List
 *              
 *              
 * Compile:     nvcc -O3 -o transpose transpose.cu -I.. [-Xptxas -dlcm=ca]
 * Run:         ./transpose
 *                  [kernel] : Select a kernel to launch
 *                  [bx] : the number of x dimension in block (default:16)
 *                  [by] : the number of y dimension in block (default:16)
 *                  [nx] : the number of rows in input matrix (default:1<<11)
 *                  [ny] : the number of columns in input matrix (default: 1<<11)
 *****************************************************************************/
#include <stdio.h>
#include <cuda_runtime.h>
#include "common/common.h"

#define BLOCKDIM_X 16
#define BLOCKDIM_Y 16

void initialData(float* in, const int size)
{
    for (int i = 0; i < size; i++) {
        in[i] = (rand() & 0xFF) / 10.f;
    }
}

void printData(float* in, const int size)
{
    for (int i = 0; i < size; i++) {
        printf("%dth element: %f\n", i, in[i]);
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

void transposeHost(float* in, float* out, const int nx, const int ny)
{
    for (int iy = 0; iy < ny; iy++) {
        for (int ix = 0; ix < nx; ix++) {
            out[ix*ny + iy] = in[iy*nx + ix];
        }
    }
}

__global__
void warmup(float* in, float* out, const int nx, const int ny)
{
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

    if (ix < nx && iy < ny)
        out[iy*nx + ix] = in[iy*nx + ix];
}

// case 0 copy kernel: access data in rows
__global__
void copyRow(float* in, float* out, const int nx, const int ny)
{
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

    if (ix < nx && iy < ny)
        out[iy*nx + ix] = in[iy*nx + ix];
}

// case 1 copy kernel: access data in columns
__global__
void copyCol(float* in, float* out, const int nx, const int ny)
{
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

    if (ix < nx && iy < ny)
        out[ix*ny + iy] = in[ix*ny + iy];
}

// case 2 transpose kernel: read in rows and write in columns
__global__
void transposeNaiveRow(float* in, float* out, const int nx, const int ny)
{
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

    if (ix < nx && iy < ny)
        out[ix*ny + iy] = in[iy*nx + ix];
}

// case 3 transpose kernel: read in columns and write in rows
__global__
void transposeNaiveCol(float* in, float* out, const int nx, const int ny)
{
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

    if (ix < nx && iy < ny)
        out[iy*nx + ix] = in[ix*ny + iy];
}

// case 4 transpose kernel: read in rows and write in columns + unroll 4 blocks
__global__
void transposeUnroll4Row(float* in, float* out, const int nx, const int ny)
{
    unsigned int ix = blockDim.x * blockIdx.x * 4 + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

    unsigned int ti = iy * nx + ix;
    unsigned int to = ix * ny + iy;

    if (ix + 3 * blockDim.x < nx && iy < ny) {
        out[to] = in[ti];
        out[to + ny * blockDim.x] = in[ti + blockDim.x];
        out[to + ny * 2 * blockDim.x] = in[ti + 2 * blockDim.x];
        out[to + ny * 3 * blockDim.x] = in[ti + 3 * blockDim.x];
    }
}

// case 5 transpose kernel: read in columns and write in rows + unroll 4 blocks
__global__
void transposeUnroll4Col(float* in, float* out, const int nx, const int ny)
{
    unsigned int ix = blockDim.x * blockIdx.x * 4 + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

    unsigned int ti = iy * nx + ix; // access in rows
    unsigned int to = ix * ny + iy; // access in columns

    if (ix + 3 * blockDim.x < nx && iy < ny) {
        out[ti] = in[to];
        out[ti + blockDim.x] = in[to + blockDim.x * ny];
        out[ti + 2 * blockDim.x] = in[to + 2 * blockDim.x * ny];
        out[ti + 3 * blockDim.x] = in[to + 3 * blockDim.x * ny];
    }
}

// case 6 transpose kernel: read in rows and write in colunms + diagonal coordinate transform
__global__
void transposeDiagonalRow(float* in, float* out, const int nx, const int ny)
{
    unsigned int blk_y = blockIdx.x;
    unsigned int blk_x = (blockIdx.x + blockIdx.y) % gridDim.x;

    unsigned int ix = blockDim.x * blk_x + threadIdx.x;
    unsigned int iy = blockDim.y * blk_y + threadIdx.y;

    if (ix < nx && iy < ny) {
        out[ix * ny + iy] = in[iy * nx + ix];
    }
}

// case 7 transpose kernel: read in columns and write in row + diagonal coordinate transform.
__global__ void transposeDiagonalCol(float* in, float* out, const int nx, const int ny)
{
    unsigned int blk_y = blockIdx.x;
    unsigned int blk_x = (blockIdx.x + blockIdx.y) % gridDim.x;

    unsigned int ix = blockDim.x * blk_x + threadIdx.x;
    unsigned int iy = blockDim.y * blk_y + threadIdx.y;

    if (ix < nx && iy < ny) {
        out[iy * nx + ix] = in[ix * ny + iy];
    }
}

int main(int argc, char** argv)
{
    // setup device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Starting transpose at device %d: %s\n", dev, deviceProp.name);
    CUDA_CHECK(cudaSetDevice(dev));

    // setup array size 2048
    int nx = 1 << 11;
    int ny = 1 << 11;

    // select a kernel and block size
    int iKernel = 0;
    int blockX = 16;
    int blockY = 16;

    if (argc > 1)
        iKernel = atoi(argv[1]);
    if (argc > 2)
        blockX = atoi(argv[2]);
    if (argc > 3)
        blockY = atoi(argv[3]);
    if (argc > 4)
        nx = atoi(argv[4]);
    if (argc > 5)
        ny = atoi(argv[5]);
    
    printf(" with matrix nx %d ny %d with kernel %d\n", nx, ny, iKernel);
    size_t nBytes = nx * ny * sizeof(float);

    // execution configuration
    dim3 blocks(blockX, blockY);
    dim3 grids((nx + blocks.x - 1) / blocks.x, (ny + blocks.y - 1) / blocks.y);

    // allocate host memory
    float *h_A = (float*)malloc(nBytes);
    float *hostRef = (float*)malloc(nBytes);
    float *gpuRef = (float*)malloc(nBytes);

    // initialize host array
    initialData(h_A, nx * ny);

    // transpose at host side
    transposeHost(h_A, hostRef, nx, ny);

    // allocate device memory
    float *d_A, *d_C;
    CUDA_CHECK(cudaMalloc((void**)&d_A, nBytes));
    CUDA_CHECK(cudaMalloc((void**)&d_C, nBytes));
    
    // copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));

    // warmup to avoid startup overhead
    warmup<<<grids, blocks>>>(d_C, d_A, nx, ny);
    CUDA_CHECK(cudaDeviceSynchronize());

    // kernel pointer and descriptor
    void (*kernel)(float*, float*, int, int);
    char *kernelName;

    // setup kernel
    switch (iKernel) {
    case 0:
        kernel = &copyRow;
        kernelName = "CopyRow       ";
        break;
        
    case 1:
        kernel = &copyCol;
        kernelName = "CopyCol       ";
        break;
        
    case 2:
        kernel = &transposeNaiveRow;
        kernelName = "NaiveRow      ";
        break;
        
    case 3:
        kernel = &transposeNaiveCol;
        kernelName = "NaiveCol      ";
        break;
        
    case 4:
        kernel = &transposeUnroll4Row;
        kernelName = "Unroll4Row    ";
        break;
        
    case 5:
        kernel = &transposeUnroll4Col;
        kernelName = "Unroll4Col    ";
        break;
        
    case 6:
        kernel = &transposeDiagonalRow;
        kernelName = "DiagonalRow   ";
        break;
        
    case 7:
        kernel = &transposeDiagonalCol;
        kernelName = "DiagonalCol   ";
        break;
    }

    // run kernel
    double start, finish;
    GET_TIME(start);
    kernel<<<grids, blocks>>>(d_A, d_C, nx, ny);
    CUDA_CHECK(cudaDeviceSynchronize());
    GET_TIME(finish);

    // calculate effective bandwidth
    float bnd = 2 * nx * ny * sizeof(float) / 1e9 / (finish-start);
    printf("%s elased %f sec <<< grid (%d,%d) block (%d,%d) >>> effective bandwidth %f GB\n",
            kernelName, finish-start, grids.x, grids.y, blocks.x, blocks.y, bnd);
    
    // check kernel result
    if (iKernel > 1) {
        CUDA_CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
        checkResult(hostRef, gpuRef, nx * ny);
    }

    // free host and device memory
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A);
    free(hostRef);
    free(gpuRef);

    CUDA_CHECK(cudaDeviceReset());
    return 0;
}