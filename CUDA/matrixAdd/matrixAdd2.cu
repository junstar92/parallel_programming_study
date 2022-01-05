/*****************************************************************************
 * File:        matrixAdd2.cu
 * Description: Matrix addition, C = A + B
 *              A,B and C have 2^14 x 2^14 dimensions.
 *              
 * Compile:     nvcc -O3 -o matrixAdd2 matrixAdd2.cu -I..
 * Run:         ./matrixAdd2 
 *****************************************************************************/
#include <stdio.h>
#include <common/common.h>
#include <cuda_runtime.h>

void initialData(float* p, const int size)
{
    for (int i = 0; i < size; i++) {
        p[i] = (float)(rand() & 0xFF) / 10.0f;
    }
}

void sumMatrixOnHost(float* A, float* B, float* C, const int nx, const int ny)
{
    for (int y = 0; y < ny; y++) {
        for (int x = 0; x < nx; x++) {
            C[y*nx + x] = A[y*nx + x] + B[y*nx + x];
        }
    }
}

void checkResult(float* hostRef, float* gpuRef, const int N)
{
    double epsilon = 1.0e-8;

    for (int i = 0; i < N; i++) {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon) {
            printf("host %f gpu %f ", hostRef[i], gpuRef[i]);
            printf("Arrays do not match.\n\n");
            break;
        }
    }
}

// grid 2D block 2D
__global__
void sumMatrixOnGPU2D(float* A, float* B, float* C, int NX, int NY)
{
    unsigned int ix = blockDim.x*blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y*blockIdx.y + threadIdx.y;
    unsigned int idx = iy*NX + ix;

    if (ix < NX && iy < NY) {
        C[idx] = A[idx] + B[idx];
    }
}

int main(int argc, char** argv)
{
    // setup device
    int dev = 0;
    cudaDeviceProp devProp;
    CUDA_CHECK(cudaGetDeviceProperties(&devProp, dev));
    CUDA_CHECK(cudaSetDevice(dev));

    // setup data size of matrix
    int nx = 1 << 14;
    int ny = 1 << 14;
    int nxy = nx * ny;
    int nBytes = nxy * sizeof(float);

    // malloc host memory
    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A = (float*)malloc(nBytes);
    h_B = (float*)malloc(nBytes);
    hostRef = (float*)malloc(nBytes);
    gpuRef = (float*)malloc(nBytes);

    // initialize data at host
    initialData(h_A, nxy);
    initialData(h_B, nxy);
    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    double start, finish;
    // add matrix at host for result
    GET_TIME(start);
    sumMatrixOnHost(h_A, h_B, hostRef, nx, ny);
    GET_TIME(finish);
    //printf("sumMatrixOnHost elapsed %f ms\n", (finish-start)*1000.f);

    // malloc device global memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void**)&d_A, nBytes));
    CUDA_CHECK(cudaMalloc((void**)&d_B, nBytes));
    CUDA_CHECK(cudaMalloc((void**)&d_C, nBytes));

    // transfer data from host to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice));

    // invoke kernel at host
    int dimx = 32;
    int dimy = 32;

    if (argc > 2) {
        dimx = atoi(argv[1]);
        dimy = atoi(argv[2]);
    }

    dim3 block(dimx, dimy);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    // execute the kernel
    CUDA_CHECK(cudaDeviceSynchronize());
    GET_TIME(start);
    sumMatrixOnGPU2D<<<grid, block>>>(d_A, d_B, d_C, nx, ny);
    CUDA_CHECK(cudaDeviceSynchronize());
    GET_TIME(finish);
    printf("sumMatrixOnGPU2D <<<(%d,%d), (%d,%d)>>> elapsed %f ms\n", grid.x, grid.y, block.x, block.y, (finish-start)*1000.f);
    CUDA_CHECK(cudaGetLastError());

    // copy kernel result back to host
    CUDA_CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    // check device result
    checkResult(hostRef, gpuRef, nxy);

    // free device global memory
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    // free host memory
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    // reset device
    CUDA_CHECK(cudaDeviceReset());

    return 0;
}