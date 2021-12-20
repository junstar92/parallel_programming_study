/*****************************************************************************
 * File:        histogram.cu
 * Description: Implement text histogram computation
 *              
 * Compile:     nvcc -o histogram histogram.cu -I..
 * Run:         ./histogram
 * Argument:
 *      "--n=<N>"           : Specify the number of elements (default: 1<<24)
 *      "--threads=<N>"     : Specify the number of threads per block (default: 256)
 *      "--blocks=<N>"      : Specify the number of blocks per grid (default: 256)
 *      "--kernel=<N>"      : Specify which kernel to run (default 0)
 *          [0] : Sequential Histogram
 *          [1] : Simple Parallel Histogram
 *          [2] : Fix [1] Kernel for memory coalescing
 *          [3] : Privatized Histogram Kernel
 *          [4] : Privatized Aggregation Histogram Kernel
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
__global__ void histo_privatized_kernel(char* data, int n, int* histo, int n_bins);
__global__ void histo_privatized_aggregation_kernel(char* data, int n, int* histo, int n_bins);

int main(int argc, char** argv)
{
    printf("[Histogram...]\n\n");

    int n = 1 << 24;
    int whichKernel = 0;
    int threads = 256;
    int blocks = 256;
    int n_bins = 7; // the number of bins, a-d, e-h, i-l, m-p, q-t, u-x, y-z

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
    h_histo = (int*)malloc(n_bins*sizeof(int));

    // init
    std::default_random_engine generator;
    std::uniform_int_distribution<char> dist('a', 'z');
    for (int i = 0; i < n; i++) {
        h_data[i] = dist(generator);
    }
    for (int i = 0; i < n_bins; i++)
        h_histo[i] = 0;

    // allocate device memory
    char* d_data;
    int *d_histo;
    CUDA_CHECK(cudaMalloc((void**)&d_data, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_histo, n_bins*sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));

    printf("Implement [Kernel %d]\n\n", whichKernel);
    double start, finish;
    GET_TIME(start);
    if (whichKernel == 0) {
        sequential_Histogram(h_data, n, h_histo);
    }
    else if (whichKernel == 1) {
        histo_kernel<<<blocks, threads>>>(d_data, n, d_histo);
        CUDA_CHECK(cudaMemcpy(h_histo, d_histo, n_bins*sizeof(int), cudaMemcpyDeviceToHost));
    }
    else if (whichKernel == 2) {
        histo_kernel_2<<<blocks, threads>>>(d_data, n, d_histo);
        CUDA_CHECK(cudaMemcpy(h_histo, d_histo, n_bins*sizeof(int), cudaMemcpyDeviceToHost));
    }
    else if (whichKernel == 3) {
        int smem_size = 2*n_bins*sizeof(int);
        histo_privatized_kernel<<<blocks, threads, smem_size>>>(d_data, n, d_histo, 7);
        CUDA_CHECK(cudaMemcpy(h_histo, d_histo, n_bins*sizeof(int), cudaMemcpyDeviceToHost));
    }
    else if (whichKernel == 4) {
        int smem_size = 2*n_bins*sizeof(int);
        histo_privatized_aggregation_kernel<<<blocks, threads, smem_size>>>(d_data, n, d_histo, 7);
        CUDA_CHECK(cudaMemcpy(h_histo, d_histo, n_bins*sizeof(int), cudaMemcpyDeviceToHost));
    }
    GET_TIME(finish);

    int total_count = 0;
    printf("histo: ");
    for (int i = 0; i < n_bins; i++) {
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

__global__
void histo_privatized_kernel(char* data, int n, int* histo, int n_bins)
{
    int tid = blockDim.x*blockIdx.x + threadIdx.x;

    // Privatized bins
    extern __shared__ int histo_s[];
    if (threadIdx.x < n_bins)
        histo_s[threadIdx.x] = 0u;
    __syncthreads();

    // histogram
    for (int i = tid; i < n; i += blockDim.x*gridDim.x) {
        int alphabet_pos = data[i] - 'a';
        if (alphabet_pos >= 0 && alphabet_pos < 26)
            atomicAdd(&histo_s[alphabet_pos/4], 1);
    }
    __syncthreads();

    // commit to global memory
    if (threadIdx.x < n_bins) {
        atomicAdd(&histo[threadIdx.x], histo_s[threadIdx.x]);
    }
}

__global__
void histo_privatized_aggregation_kernel(char* data, int n, int* histo, int n_bins)
{
    int tid = blockDim.x*blockIdx.x + threadIdx.x;

    // Privatized bins
    extern __shared__ int histo_s[];
    if (threadIdx.x < n_bins)
        histo_s[threadIdx.x] = 0u;
    __syncthreads();

    int prev_index = -1;
    int accumulator = 0;

    // histogram
    for (int i = tid; i < n; i += blockDim.x*gridDim.x) {
        int alphabet_pos = data[i] - 'a';
        if (alphabet_pos >= 0 && alphabet_pos < 26) {
            int curr_index = alphabet_pos/4;
            if (curr_index != prev_index) {
                if (prev_index != -1 && accumulator > 0)
                    atomicAdd(&histo_s[prev_index], accumulator);
                accumulator = 1;
                prev_index = curr_index;
            }
            else {
                accumulator++;
            }
        }
    }
    if (accumulator > 0)
        atomicAdd(&histo_s[prev_index], accumulator);
    __syncthreads();

    // commit to global memory
    if (threadIdx.x < n_bins) {
        atomicAdd(&histo[threadIdx.x], histo_s[threadIdx.x]);
    }
}