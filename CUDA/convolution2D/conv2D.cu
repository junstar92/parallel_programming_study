/*****************************************************************************
 * File:        conv2D.cu
 * Description: Implement 2D convolution
 *              
 * Compile:     nvcc -o conv2D conv2D.cu -I..
 * Run:         ./conv2D
 * Argument:
 *      "--width=<N>"       : Specify the number of width of input image (default: 1080)
 *      "--height=<N>"      : Specify the number of height of input image (default: 1920)
 *      "--channel=<N>"     : Specify the number of channels of input image (default: 1, <= 3)
 *      "--filter=<N>"      : Specify the number of filter width for convolution (default: 5)
 *      "--kernel=<N>"      : Specify which kernel to run (default 0)
 *          [0] : basic 2D convolution with constant memory
 *          [1] : tiled 2D convolution with constant memory
 *****************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#include <common/common.h>
#include <common/common_string.h>

#define O_TILE_WIDTH 16
#define MAX_KERNEL_WIDTH 10

__constant__ float M[MAX_KERNEL_WIDTH * MAX_KERNEL_WIDTH];
void convolution2D_CPU(float* in, float* out, float* kernel, int width, int height, int channels, int kernel_width);
__global__ void convolution2D(float* in, float* out, int width, int height, int channels, int kernel_width);
__global__ void convolution2D_tiled(float* P, float* N, int width, int height, int channels, int kernel_width);
bool run(int width, int height, int channels, int kernel_width, int whichKernel);

int main(int argc, char** argv)
{
    printf("[2D Convolution...]\n\n");

    int height = 1080;
    int width = 1920;
    int channels = 1;
    int kernel_width = 5;
    int whichKernel = 0;

    if (checkCmdLineFlag(argc, (const char **)argv, "width")) {
        width = getCmdLineArgumentInt(argc, (const char **)argv, "width");
    }
    if (checkCmdLineFlag(argc, (const char **)argv, "height")) {
        height = getCmdLineArgumentInt(argc, (const char **)argv, "height");
    }
    if (checkCmdLineFlag(argc, (const char **)argv, "channels")) {
        channels = getCmdLineArgumentInt(argc, (const char **)argv, "channels");
    }
    if (checkCmdLineFlag(argc, (const char **)argv, "filter")) {
        kernel_width = getCmdLineArgumentInt(argc, (const char **)argv, "filter");
    }
    if (checkCmdLineFlag(argc, (const char **)argv, "kernel")) {
        whichKernel = getCmdLineArgumentInt(argc, (const char **)argv, "kernel");
    }

    printf("Kernel Size: %d x %d\n", kernel_width, kernel_width);
    printf("Input Size: %d x %d x %d\n", width, height, channels);

    int dev = 0;
    cudaSetDevice(dev);

    bool result = run(width, height, channels, kernel_width, whichKernel);
    
    printf(result ? "Test PASSED\n" : "Test FAILED!\n");
    cudaDeviceReset();

    return 0;
}

bool run(int width, int height, int channels, int kernel_width, int whichKernel)
{
    unsigned int bytes = width * height * channels * sizeof(float);
    float* h_in, *h_out, *h_kernel;
    float* d_in, *d_out;

    // allocate host memory
    h_in = (float*)malloc(bytes);
    h_out = (float*)malloc(bytes);
    h_kernel = (float*)malloc(kernel_width*kernel_width*sizeof(float));

    // init value
    for (int c = 0; c < channels; c++)
        for (int i = 0; i < height; i++)
            for (int j = 0; j < width; j++)
                h_in[c*height*width + i*width + j] = rand() / (float)RAND_MAX;
    for (int i = 0; i < kernel_width; i++)
        for (int j = 0; j < kernel_width; j++)
            h_kernel[i*kernel_width + j] = rand() / (float)RAND_MAX;
    
    // allocate device memory
    CUDA_CHECK(cudaMalloc((void**)&d_in, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_out, bytes));

    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyToSymbol(M, h_kernel, kernel_width*kernel_width*sizeof(float)));

    // launch Kernel
    printf("\nLaunch Kernel...\n");
    double start, finish;
    if (whichKernel > 1)
        whichKernel = 0;
    if (whichKernel == 0) {
        // basic 2d conv
        dim3 dimBlock(O_TILE_WIDTH, O_TILE_WIDTH, 1);
        dim3 dimGrid((width + O_TILE_WIDTH - 1) / O_TILE_WIDTH, (height + O_TILE_WIDTH - 1) / O_TILE_WIDTH, channels);
        printf("block dim: %d x %d x %d\n", dimBlock.x, dimBlock.y, dimBlock.z);
        printf("grid dim: %d x %d x %d\n", dimGrid.x, dimGrid.y, dimGrid.z);

        //warm up
        convolution2D<<<dimGrid, dimBlock>>>(d_in, d_out, width, height, channels, kernel_width);
        CUDA_CHECK(cudaGetLastError());
        cudaDeviceSynchronize();

        GET_TIME(start);
        convolution2D<<<dimGrid, dimBlock>>>(d_in, d_out, width, height, channels, kernel_width);
        CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
        cudaDeviceSynchronize();
        GET_TIME(finish);
    }
    else {
        // tiled 2d conv
        const int I_TILE_WIDTH = O_TILE_WIDTH + kernel_width - 1;
        dim3 dimBlock(I_TILE_WIDTH, I_TILE_WIDTH, 1);
        dim3 dimGrid((width + O_TILE_WIDTH - 1) / O_TILE_WIDTH, (height + O_TILE_WIDTH - 1) / O_TILE_WIDTH, channels);
        printf("TILE size: %d\n", I_TILE_WIDTH);
        printf("block dim: %d x %d x %d\n", dimBlock.x, dimBlock.y, dimBlock.z);
        printf("grid dim: %d x %d x %d\n", dimGrid.x, dimGrid.y, dimGrid.z);

        //warm up
        convolution2D_tiled<<<dimGrid, dimBlock>>>(d_in, d_out, width, height, channels, kernel_width);
        CUDA_CHECK(cudaGetLastError());
        cudaDeviceSynchronize();

        GET_TIME(start);
        convolution2D_tiled<<<dimGrid, dimBlock>>>(d_in, d_out, width, height, channels, kernel_width);
        CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
        cudaDeviceSynchronize();
        GET_TIME(finish);
    }

    // result in CPU
    float* cpu_out = (float*)malloc(bytes);
    printf("\nCalculating in CPU...\n");
    convolution2D_CPU(h_in, cpu_out, h_kernel, width, height, channels, kernel_width);
    
    int precision = 8;
    double threshold = 1e-8 * channels*width*height;
    double diff = 0.0;
    for (int i = 0; i < channels*width*height; i++) {
        diff += fabs((double)cpu_out[i] - (double)h_out[i]);
    }
    diff /= (double)channels*width*height;

    printf("[Kernel %d] Throughput = %.4f GB/s, Time = %.5f ms\n",
        whichKernel, ((double)bytes / (finish-start))*1.0e-9, (finish-start)*1000);
    printf("Error : %.*f (threshold: %f)\n", precision, (double)diff, threshold);

    // free memory
    free(h_in);
    free(h_out);
    free(h_kernel);
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    free(cpu_out);
    
    return (diff < threshold);
}

void convolution2D_CPU(float* in, float* out, float* kernel, int width, int height, int channels, int kernel_width)
{
    for (int ch = 0; ch < channels; ch++) {
        for (int row = 0; row < height; row++) {
            for (int col = 0; col < width; col++) {
                int row_start_point = row - kernel_width/2;
                int col_start_point = col - kernel_width/2;
                float val = 0.f;

                for (int i = 0; i < kernel_width; i++) {
                    for (int j = 0; j < kernel_width; j++) {
                        int row_idx = row_start_point + i;
                        int col_idx = col_start_point + j;

                        if (row_idx >= 0 && row_idx < height && col_idx >= 0 && col_idx < width) {
                            val += in[ch*width*height + row_idx*width + col_idx]*kernel[i*kernel_width + j];
                        }
                    }
                }

                out[ch*width*height + row*width + col] = val;
            }
        }
    }
}

__global__
void convolution2D(float* in, float* out, int width, int height, int channels, int kernel_width)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;

    int ch = blockDim.z*blockIdx.z + tz;
    int row = blockDim.y*blockIdx.y + ty;
    int col = blockDim.x*blockIdx.x + tx;

    if (col < width && row < height && ch < channels) {
        int row_start_point = row - kernel_width/2;
        int col_start_point = col - kernel_width/2;
        float val = 0.f;

        for (int i = 0; i < kernel_width; i++) {
            for (int j = 0; j < kernel_width; j++) {
                int row_idx = row_start_point + i;
                int col_idx = col_start_point + j;
                if (row_idx >= 0 && row_idx < height && col_idx >= 0 && col_idx < width) {
                    val += in[ch*width*height + row_idx*width + col_idx]*M[i*kernel_width + j];
                }
            }
        }

        out[ch*width*height + row*width + col] = val;
    }
}

__global__
void convolution2D_tiled(float* in, float* out, int width, int height, int channels, int kernel_width)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    int ch = blockDim.z*blockIdx.z + tz;
    int row_o = blockIdx.y*O_TILE_WIDTH + ty;
    int col_o = blockIdx.x*O_TILE_WIDTH + tx;
    int row_i = row_o - kernel_width/2;
    int col_i = col_o - kernel_width/2;

    __shared__ float in_tile[O_TILE_WIDTH + MAX_KERNEL_WIDTH - 1][O_TILE_WIDTH + MAX_KERNEL_WIDTH - 1];
    if ((row_i >= 0) && (row_i < height) && (col_i >= 0) && (col_i < width)) {
        in_tile[ty][tx] = in[ch*width*height + row_i*width + col_i];
    }
    else {
        in_tile[ty][tx] = 0;
    }

    __syncthreads();

    float val = 0.f;
    if (ty < O_TILE_WIDTH && tx < O_TILE_WIDTH && ch < channels) {
        for (int i = 0; i < kernel_width; i++) {
            for (int j = 0; j < kernel_width; j++) {
                val += in_tile[i+ty][j+tx] * M[i*kernel_width + j];
            }
        }

        if (row_o < height && col_o < width)
            out[ch*width*height + row_o*width + col_o] = val;
    }
}