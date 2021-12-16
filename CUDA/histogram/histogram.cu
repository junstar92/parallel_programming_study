/*****************************************************************************
 * File:        histogram.cu
 * Description: 
 *              
 * Compile:     nvcc -o histogram histogram.cu -I..
 * Run:         ./histogram
 * Argument:
 *      "--n=<N>"           : Specify the number of elements (default: 1<<24)
 *      "--kernel=<N>"      : Specify which kernel to run (default 0)
 *      "--threads=<N>"     : Specify the number of threads per block (default: 256)
 *      "--blocks=<N>"      : Specify the number of blocks per grid (default: 256)
 *          [0] : Sequential Histogram
 *          [1] : Simple Parallel Histogram
 *          [2] : Fix [1] Kernel for memory coalescing
 *****************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <random>
#include <cuda_runtime.h>

#include <common/common.h>
#include <common/common_string.h>

void sequential_Histogram(char* data, int n, int* histo);
__global__ void histo_kernel(char* data, int n, int* histo);
__global__ void histo_kernel_2(char* data, int n, int* histo);

int main(int argc, char** argv)
{
    printf("[Histogram...]\n\n");

    int n = 1 << 24;
    int whichKernel = 0;
    int threads = 256;
    int blocks = 256;

    if (checkCmdLineFlag(argc, (const char **)argv, "n")) {
        n = getCmdLineArgumentInt(argc, (const char **)argv, "n");
    }
    if (checkCmdLineFlag(argc, (const char **)argv, "kernel")) {
        whichKernel = getCmdLineArgumentInt(argc, (const char **)argv, "kernel");
    }
    if (checkCmdLineFlag(argc, (const char **)argv, "threads")) {
        threads = getCmdLineArgumentInt(argc, (const char **)argv, "threads");
    }
    if (checkCmdLineFlag(argc, (const char **)argv, "blocks")) {
        blocks = getCmdLineArgumentInt(argc, (const char **)argv, "blocks");
    }

    printf("The number of elements: %d\n", n);
    printf("Threads: %d / Blocks: %d\n\n", threads, blocks);
    
    int dev = 0;
    CUDA_CHECK(cudaSetDevice(dev));

    unsigned int bytes = n*sizeof(char);
    char* h_data;
    int* h_histo;

    // allocate host memory
    h_data = (char*)malloc(bytes);
    h_histo = (int*)malloc(7*sizeof(int));

    // init
    std::default_random_engine generator;
    std::uniform_int_distribution<char> dist('a', 'z');
    for (int i = 0; i < n; i++) {
        h_data[i] = dist(generator);
    }
    for (int i = 0; i < 7; i++)
        h_histo[i] = 0;

    // allocate device memory
    char* d_data;
    int *d_histo;
    CUDA_CHECK(cudaMalloc((void**)&d_data, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_histo, 7*sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));

    printf("Implement [Kernel %d]\n\n", whichKernel);
    double start, finish;
    GET_TIME(start);
    if (whichKernel == 0) {
        sequential_Histogram(h_data, n, h_histo);
    }
    else if (whichKernel == 1) {
        histo_kernel<<<blocks, threads>>>(d_data, n, d_histo);
        CUDA_CHECK(cudaMemcpy(h_histo, d_histo, 7*sizeof(int), cudaMemcpyDeviceToHost));
    }
    else if (whichKernel == 2) {
        histo_kernel_2<<<blocks, threads>>>(d_data, n, d_histo);
        CUDA_CHECK(cudaMemcpy(h_histo, d_histo, 7*sizeof(int), cudaMemcpyDeviceToHost));
    }
    GET_TIME(finish);

    int total_count = 0;
    printf("histo: ");
    for (int i = 0; i < 7; i++) {
        printf("%d ", h_histo[i]);
        total_count += h_histo[i];
    }
    printf("\n\n");
    printf("Total Count : %d\n", total_count);
    printf("Time: %f msec\n\n", (finish-start)*1000);

    printf(total_count == n ? "Test PASS\n" : "Test FAILED!\n");

    // free memory
    free(h_data);
    free(h_histo);
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_histo));

    return 0;
}

void sequential_Histogram(char* data, int n, int* histo)
{
    for (int i = 0; i < n; i++) {
        int alphabet_pos = data[i] - 'a';
        if (alphabet_pos >= 0 && alphabet_pos < 26)
            histo[alphabet_pos/4]++;
    }
}

__global__
void histo_kernel(char* data, int n, int* histo)
{
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    int section_size = (n - 1) / (blockDim.x * gridDim.x) + 1;
    int start = i*section_size;

    for (int k = 0; k < section_size; k++) {
        if (start + k < n) {
            int alphabet_pos = data[start+k] - 'a';
            if (alphabet_pos >= 0 && alphabet_pos < 26)
                atomicAdd(&histo[alphabet_pos/4], 1);
        }
    }
}

__global__
void histo_kernel_2(char* data, int n, int* histo)
{
    int tid = blockDim.x*blockIdx.x + threadIdx.x;

    for (int i = tid; i < n; i += blockDim.x*gridDim.x) {
        int alphabet_pos = data[i] - 'a';
        if (alphabet_pos >= 0 && alphabet_pos < 26)
            atomicAdd(&histo[alphabet_pos/4], 1);
    }
}