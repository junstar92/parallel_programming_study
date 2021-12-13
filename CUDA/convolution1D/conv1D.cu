/*****************************************************************************
 * File:        conv1D.cpp
 * Description: Implement 1D convolution
 *              
 * Compile:     nvcc -o conv1D conv1D.cu -I..
 * Run:         ./conv1D
 * Argument:
 *      "--n=<N>"           : Specify the number of elements to reduce (default: 1 << 20)
 *      "--threads=<N>"     : Specify the number of threads per block (default: 256)
 *      "--iteration=<N>"   : Specify the number of iteration (default: 100)
 *      "--filter=<N>"      : Specify the number of filter width for convolution (default: 5)
 *      "--kernel=<N>"      : Specify which kernel to run (default 1)
 *          [0] : basic 1D convolution without constant memory
 *          [1] : basic 1D convolution with constant memory
 *          [2] : tiled 1D convolution
 *          [3] : tiled 1D convolution with L2 cache
 *****************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#include <common/common.h>
#include <common/common_string.h>

#define MAX_KERNEL_WIDTH 10

__constant__ float M[MAX_KERNEL_WIDTH];

bool run(int size, int kernel_width, int threads, int blocks, int nIter, int whichKernel);
void convolution1D_CPU(float* h_N, float* h_M, float* h_P, int Kernel_Width, int Width);
__global__ void convolution1D_basic_woConMem(float* d_N, float* d_M, float* d_P, int Kernel_Width, int Width);
__global__ void convolution1D_basic_wConMem(float* d_N, float* d_P, int Kernel_Width, int Width);
template<unsigned int TILE_SIZE>
__global__ void convolution1D_tiled(float* d_N, float* d_P, int Kernel_Width, int Width);
template<unsigned int TILE_SIZE>
__global__ void convolution1D_tiled_L2(float* d_N, float* d_P, int Kernel_Width, int Width);

int main(int argc, char** argv)
{
    printf("[1D Convolution...]\n\n");

    int size = 1 << 20;
    int threads = 256;
    int nIter = 100;
    int whichKernel = 1;
    int kernel_width = 5;

    if (checkCmdLineFlag(argc, (const char **)argv, "n")) {
        size = getCmdLineArgumentInt(argc, (const char **)argv, "n");
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "threads")) {
        threads = getCmdLineArgumentInt(argc, (const char **)argv, "threads");
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "kernel")) {
        whichKernel = getCmdLineArgumentInt(argc, (const char **)argv, "kernel");
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "iteration")) {
        nIter = getCmdLineArgumentInt(argc, (const char **)argv, "iteration");
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "filter")) {
        kernel_width = getCmdLineArgumentInt(argc, (const char **)argv, "filter");
    }

    printf("Kernel Width: %d\n", kernel_width);
    printf("%d elements\n", size);
    printf("%d threads\n", threads);
    int blocks = (size + threads - 1) / threads;
    printf("%d blocks\n", blocks);

    int dev = 0;
    cudaSetDevice(dev);

    bool result = run(size, kernel_width, threads, blocks, nIter, whichKernel);

    printf(result ? "Test PASSED\n" : "Test FAILED!\n");

    return 0;
}

bool run(int size, int kernel_width, int threads, int blocks, int nIter, int whichKernel)
{
    unsigned int bytes = size * sizeof(float);
    float *h_N, *h_M, *h_P;
    float *d_N, *d_M, *d_P;

    // allocate host memory
    h_N = (float*)malloc(bytes);
    h_M = (float*)malloc(kernel_width * sizeof(float));
    h_P = (float*)malloc(bytes);

    // init value
    for (int i = 0; i < size; i++)
        h_N[i] = rand() / (float)RAND_MAX;
    for (int i = 0; i < kernel_width; i++)
        h_M[i] = rand() / (float)RAND_MAX;    

    // allocate device memory
    CUDA_CHECK(cudaMalloc((void**)&d_N, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_M, kernel_width * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_P, bytes));

    CUDA_CHECK(cudaMemcpy(d_N, h_N, bytes, cudaMemcpyHostToDevice));
    if (whichKernel == 0) {
        CUDA_CHECK(cudaMemcpy(d_M, h_M, kernel_width * sizeof(float), cudaMemcpyHostToDevice));
    }
    else {
        CUDA_CHECK(cudaMemcpyToSymbol(M, h_M, kernel_width * sizeof(float)));
    }

    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);
    
    // warm up
    convolution1D_basic_woConMem<<<dimGrid, dimBlock>>>(d_N, d_M, d_P, kernel_width, size);

    double start, finish, total_time = 0.f;
    for (int i = 0; i < nIter; i++) {
        cudaDeviceSynchronize();
        GET_TIME(start);
        switch (whichKernel) {
            case 0:
                convolution1D_basic_woConMem<<<dimGrid, dimBlock>>>(d_N, d_M, d_P, kernel_width, size);
                break;
            default:
            case 1:
                convolution1D_basic_wConMem<<<dimGrid, dimBlock>>>(d_N, d_P, kernel_width, size);
                break;
            case 2:
                switch (threads) {
                    case 1024:
                        convolution1D_tiled<1024><<<dimGrid, dimBlock>>>(d_N, d_P, kernel_width, size);
                        break;
                    case 512:
                        convolution1D_tiled<512><<<dimGrid, dimBlock>>>(d_N, d_P, kernel_width, size);
                        break;
                    case 256:
                        convolution1D_tiled<256><<<dimGrid, dimBlock>>>(d_N, d_P, kernel_width, size);
                        break;
                    case 128:
                        convolution1D_tiled<128><<<dimGrid, dimBlock>>>(d_N, d_P, kernel_width, size);
                        break;
                    case 64:
                        convolution1D_tiled<64><<<dimGrid, dimBlock>>>(d_N, d_P, kernel_width, size);
                        break;
                    case 32:
                        convolution1D_tiled<32><<<dimGrid, dimBlock>>>(d_N, d_P, kernel_width, size);
                        break;
                    case 16:
                        convolution1D_tiled<16><<<dimGrid, dimBlock>>>(d_N, d_P, kernel_width, size);
                        break;
                    case 8:
                        convolution1D_tiled<8><<<dimGrid, dimBlock>>>(d_N, d_P, kernel_width, size);
                        break;
                    case 4:
                        convolution1D_tiled<4><<<dimGrid, dimBlock>>>(d_N, d_P, kernel_width, size);
                        break;
                    case 2:
                        convolution1D_tiled<2><<<dimGrid, dimBlock>>>(d_N, d_P, kernel_width, size);
                        break;
                    case 1:
                        convolution1D_tiled<1><<<dimGrid, dimBlock>>>(d_N, d_P, kernel_width, size);
                        break;
                }
                break;
            case 3:
                switch (threads) {
                    case 1024:
                        convolution1D_tiled_L2<1024><<<dimGrid, dimBlock>>>(d_N, d_P, kernel_width, size);
                        break;
                    case 512:
                        convolution1D_tiled_L2<512><<<dimGrid, dimBlock>>>(d_N, d_P, kernel_width, size);
                        break;
                    case 256:
                        convolution1D_tiled_L2<256><<<dimGrid, dimBlock>>>(d_N, d_P, kernel_width, size);
                        break;
                    case 128:
                        convolution1D_tiled_L2<128><<<dimGrid, dimBlock>>>(d_N, d_P, kernel_width, size);
                        break;
                    case 64:
                        convolution1D_tiled_L2<64><<<dimGrid, dimBlock>>>(d_N, d_P, kernel_width, size);
                        break;
                    case 32:
                        convolution1D_tiled_L2<32><<<dimGrid, dimBlock>>>(d_N, d_P, kernel_width, size);
                        break;
                    case 16:
                        convolution1D_tiled_L2<16><<<dimGrid, dimBlock>>>(d_N, d_P, kernel_width, size);
                        break;
                    case 8:
                        convolution1D_tiled_L2<8><<<dimGrid, dimBlock>>>(d_N, d_P, kernel_width, size);
                        break;
                    case 4:
                        convolution1D_tiled_L2<4><<<dimGrid, dimBlock>>>(d_N, d_P, kernel_width, size);
                        break;
                    case 2:
                        convolution1D_tiled_L2<2><<<dimGrid, dimBlock>>>(d_N, d_P, kernel_width, size);
                        break;
                    case 1:
                        convolution1D_tiled_L2<1><<<dimGrid, dimBlock>>>(d_N, d_P, kernel_width, size);
                        break;
                }
                break;
        }

        CUDA_CHECK(cudaMemcpy(h_P, d_P, bytes, cudaMemcpyDeviceToHost));
        cudaDeviceSynchronize();
        GET_TIME(finish);

        total_time += (finish - start);
    }

    // result in CPU
    float* cpu_P = (float*)malloc(bytes);
    convolution1D_CPU(h_N, h_M, cpu_P, kernel_width, size);

    int precision = 8;
    double threshold = 1e-8 * size;
    double diff = 0.0;
    for (int i = 0; i < size; i++) {
        diff += fabs((double)cpu_P[i] - (double)h_P[i]);
    }
    diff /= (double)size;

    double elapsedTime = (total_time / (double)nIter);
    printf("[Kernel %d] Throghput = %.4f GB/s, Time = %.5f ms\n",
        whichKernel, ((double)bytes / elapsedTime)*1.0e-9, elapsedTime * 1000);
    printf("Error : %.*f\n", precision, (double)diff);

    // free memory
    free(h_P);
    free(h_M);
    free(h_N);
    CUDA_CHECK(cudaFree(d_P));
    CUDA_CHECK(cudaFree(d_M));
    CUDA_CHECK(cudaFree(d_N));
    free(cpu_P);

    return (diff < threshold);
}

void convolution1D_CPU(float* h_N, float* h_M, float* h_P, int Kernel_Width, int Width)
{
    for (int i = 0; i < Width; i++) {
        float Pvalue = 0.f;
        int N_start_point = i - (Kernel_Width/2);
        for (int j = 0; j < Kernel_Width; j++) {
            if (N_start_point + j >= 0 && N_start_point + j < Width)
                Pvalue += h_N[N_start_point + j] * h_M[j];
        }
        h_P[i] = Pvalue;
    }
}

__global__
void convolution1D_basic_woConMem(float* d_N, float* d_M, float* d_P, int Kernel_Width, int Width)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;

    float Pvalue = 0.f;
    int N_start_point = i - (Kernel_Width/2);
    for (int j = 0; j < Kernel_Width; j++) {
        if (N_start_point + j >= 0 && N_start_point + j < Width)
            Pvalue += d_N[N_start_point + j] * d_M[j];
    }

    if (i < Width)
        d_P[i] = Pvalue;
}
__global__
void convolution1D_basic_wConMem(float* d_N, float* d_P, int Kernel_Width, int Width)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;

    float Pvalue = 0.f;
    int N_start_point = i - (Kernel_Width/2);
    for (int j = 0; j < Kernel_Width; j++) {
        if (N_start_point + j >= 0 && N_start_point + j < Width)
            Pvalue += d_N[N_start_point + j] * M[j];
    }

    if (i < Width)
        d_P[i] = Pvalue;
}

template<unsigned int TILE_SIZE>
__global__
void convolution1D_tiled(float* d_N, float* d_P, int Kernel_Width, int Width)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    __shared__ float d_Nds[TILE_SIZE + MAX_KERNEL_WIDTH - 1];
    int n = Kernel_Width / 2;

    if (threadIdx.x >= blockDim.x - n) {
        int halo_index_left = (blockIdx.x - 1)*blockDim.x + threadIdx.x;
        d_Nds[threadIdx.x - (blockDim.x - n)] = (halo_index_left < 0) ? 0 : d_N[halo_index_left];
    }

    d_Nds[n + threadIdx.x] = d_N[i];
    
    if (threadIdx.x < n) {
        int halo_index_right = (blockIdx.x + 1)*blockDim.x + threadIdx.x;
        d_Nds[n + blockDim.x + threadIdx.x] = (halo_index_right >= Width) ? 0 : d_N[halo_index_right];
    }

    __syncthreads();

    float Pvalue = 0;
    for (int j = 0; j < Kernel_Width; j++)
        Pvalue += d_Nds[threadIdx.x + j]*M[j];
    d_P[i] = Pvalue;
}

template<unsigned int TILE_SIZE>
__global__
void convolution1D_tiled_L2(float* d_N, float* d_P, int Kernel_Width, int Width)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    __shared__ float d_Nds[TILE_SIZE];

    d_Nds[threadIdx.x] = d_N[i];
    __syncthreads();

    int this_tile_start_point = blockIdx.x * blockDim.x;
    int next_tile_start_point = (blockIdx.x + 1) * blockDim.x;
    int N_start_point = i - (Kernel_Width/2);
    float Pvalue = 0;
    for (int j = 0; j < Kernel_Width; j++) {
        int N_index = N_start_point + j;
        if (N_index >= 0 && N_index < Width) {
            if ((N_index >= this_tile_start_point) && (N_index < next_tile_start_point)) {
                Pvalue += d_Nds[threadIdx.x + j - (Kernel_Width/2)]*M[j];
            }
            else {
                Pvalue += d_N[N_index] * M[j];
            }
        }
    }
    d_P[i] = Pvalue;
}