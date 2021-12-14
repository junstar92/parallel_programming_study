/*****************************************************************************
 * File:        prefixSum.cu
 * Description: Implement prefix summation
 *              
 * Compile:     nvcc -o prefixSum prefixSum.cu -I..
 * Run:         ./prefixSum
 * Argument:
 *      "--n=<N>"           : Specify the number of elements (default: 1<<24)
 *      "--threads=<N>"     : Specify the number of block's threads (default: 256)
 *      "--kernel=<N>"      : Specify which kernel to run (default 0)
 *          [0] : Kogge-Stone Scan Algorithm
 *          [1] : Brent-Kung Adder Algorithm (todo)
 *****************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#include <common/common.h>
#include <common/common_string.h>

#define SECTION_SIZE 512

bool run(int n, int threads, int whichKernel);
void launchKernel(int n, int whichKernel, dim3 dimBlock, dim3 dimGrid, float* h_out, float* d_in, float* d_out);
void sequentialScan(float* x, float* y, int n);
__global__ void koggeStoneScan(float* X, float* Y, int n);

int main(int argc, char** argv)
{
    printf("[Prefix Summation...]\n\n");
    int n = 1 << 24;
    int threads = 256;
    int whichKernel = 0;

    if (checkCmdLineFlag(argc, (const char **)argv, "n")) {
        n = getCmdLineArgumentInt(argc, (const char **)argv, "n");
    }
    if (checkCmdLineFlag(argc, (const char **)argv, "threads")) {
        threads = getCmdLineArgumentInt(argc, (const char **)argv, "threads");
    }
    if (checkCmdLineFlag(argc, (const char **)argv, "kernel")) {
        whichKernel = getCmdLineArgumentInt(argc, (const char **)argv, "kernel");
    }

    printf("The number of elements: %d\n", n);
    
    int dev = 0;
    CUDA_CHECK(cudaSetDevice(dev));

    run(n, threads, whichKernel);

    printf("[Done]\n\n");
    cudaDeviceReset();

    return 0;
}

bool run(int n, int threads, int whichKernel)
{
    unsigned int bytes = n * sizeof(float);
    float* h_in, *h_out;
    float* d_in, *d_out;

    // allocate host memory
    h_in = (float*)malloc(bytes);
    h_out = (float*)malloc(bytes);

    // init value
    for (int i = 0; i < n; i++)
        h_in[i] = rand() / (float)RAND_MAX;

    // allocate device memory
    CUDA_CHECK(cudaMalloc((void**)&d_in, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_out, bytes));

    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    double start, finish, total_time = 0.f;
    // Launch Kernel
    dim3 dimBlock(SECTION_SIZE, 1, 1);
    dim3 dimGrid((n + SECTION_SIZE - 1) / SECTION_SIZE, 1, 1);
    printf("block dim: %d x %d x %d\n", dimBlock.x, dimBlock.y, dimBlock.z);
    printf("grid dim: %d x %d x %d\n\n", dimGrid.x, dimGrid.y, dimGrid.z);

    //warm up
    launchKernel(n, whichKernel, dimBlock, dimGrid, h_out, d_in, d_out);

    // launching
    int nIter = 100;
    for (int i = 0; i < nIter; i++) {
        cudaDeviceSynchronize();
        GET_TIME(start);
        launchKernel(n, whichKernel, dimBlock, dimGrid, h_out, d_in, d_out);
        cudaDeviceSynchronize();
        GET_TIME(finish);
        total_time += (finish - start);
    }
    double avg_time = total_time / (double)nIter;
    printf("[Kernel %d] Throughput = %.4f GB/s, Time = %.5f ms\n\n", 
            whichKernel, ((double)bytes / avg_time)*1.0e-9, avg_time*1000);
    
    printf("[Sequtial Scan...]\n");
    float* cpu_out = (float*)malloc(bytes);
    GET_TIME(start);
    sequentialScan(h_in, cpu_out, n);
    GET_TIME(finish);
    printf("[Sequential] Throughput = %.4f GB/s, Time = %.5f ms\n\n", 
            ((double)bytes / (finish-start))*1.0e-9, (finish-start)*1000);

    // verification
    int precision = 8;
    double threshold = 1e-8 * n;
    double diff = 0.f;
    for (int i = 0; i < n; i++) {
        diff += fabs((double)cpu_out[i] - (double)h_out[i]);
    }
    diff /= (double)n;
    printf("Error = %.*f\n", precision, (double)diff);

    free(h_in);
    free(h_out);
    free(cpu_out);
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));

    return diff < threshold;
}

void launchKernel(int n, int whichKernel, dim3 dimBlock, dim3 dimGrid, float* h_out, float* d_in, float* d_out)
{
    switch (whichKernel) {
        default:
        case 0:
            koggeStoneScan<<<dimGrid, dimBlock>>>(d_in, d_out, n);
            CUDA_CHECK(cudaMemcpy(h_out, d_out, n*sizeof(float), cudaMemcpyDeviceToHost));
            for (int i = 1; i < dimGrid.x; i++) {
                int offset = i*SECTION_SIZE;
                float tmp = h_out[offset - 1];
                
                for (int j = 0; (j < SECTION_SIZE) && (offset + j < n); j++) {
                    h_out[offset + j] += tmp;
                }
            }
            break;
    }
}

void sequentialScan(float* X, float* Y, int n)
{
    //float accumulator = X[0];
    Y[0] = X[0];
    for (int i = 1; i < n; i++) {
        Y[i] = Y[i-1]+X[i];
    }
}

__global__
void koggeStoneScan(float* X, float* Y, int n)
{
    __shared__ float XY[SECTION_SIZE];
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    XY[threadIdx.x] = (i < n) ? X[i] : 0;

    float temp = 0.f;
    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        __syncthreads();
        if (threadIdx.x >= stride) {
            temp = XY[threadIdx.x] + XY[threadIdx.x - stride];
        }
        __syncthreads();
        if (threadIdx.x >= stride)
            XY[threadIdx.x] = temp;
    }
    
    if (i < n)
        Y[i] = XY[threadIdx.x];
}