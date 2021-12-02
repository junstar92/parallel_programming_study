/*****************************************************************************
 * File:        matrixAdd.cu
 * Description: Matrix addition, C = A + B
 *              A,B and C have m x n dimensions.
 *              
 * Compile:     nvcc -o matrixAdd matrixAdd.cu -I.. -lcuda
 * Run:         ./matrixAdd <m> <n>
 *                  <m> : the number of rows in Matrix A, B
 *                  <n> : the number of columns in Matrix A, B.
 *****************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <common/common.h>

void Usage(char prog_name[]);
__global__ void matrixAdd(const float *A, const float *B, float *C, const int M, const int N);

int main(int argc, char* argv[])
{
    if (argc != 3) {
        Usage(argv[0]);
    }

    int m = strtol(argv[1], NULL, 10);
    int n = strtol(argv[2], NULL, 10);
    printf("[Matrix addition, C = A + B]\n");
    printf("\tA, B, and C are (%d x %d) matrix\n", m, n);

    // Allocate the host matrix A, B, C
    float *h_A = (float*)malloc(m * n * sizeof(float));
    float *h_B = (float*)malloc(m * n * sizeof(float));
    float *h_C = (float*)malloc(m * n * sizeof(float));

    // Verify that allocations succeeded
    if (h_A == NULL || h_B == NULL || h_C == NULL) {
        fprintf(stderr, "Failed to allocate host matrix!\n");
        exit(EXIT_SUCCESS);
    }
    
    // Initialize that host matrix
    common_random_init_matrix<float>(h_A, m, n);
    common_random_init_matrix<float>(h_B, m, n);

    // Allocate the device matrix A, B, C
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void**)&d_A, m * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_B, m * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_C, m * n * sizeof(float)));

    // Copy the host input matrix A and B in host memory 
    // to the device input matrix in device memory
    printf("Copy input data from the host memory to the CUDA device\n");
    CUDA_CHECK(cudaMemcpy(d_A, h_A, m * n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, m * n * sizeof(float), cudaMemcpyHostToDevice));

    // Allocate CUDA events for estimating
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Launch the Matrix Add CUDA Kernel
    const int block_size = 16;
    dim3 threads(block_size, block_size);
    dim3 grid(ceil(m / (float)threads.x), ceil(n / (float)threads.y));
    printf("CUDA kernel launch with (%d x %d) blocks of (%d x %d) threads\n", grid.x, grid.y, threads.x, threads.y);

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(start));

    matrixAdd<<<grid, threads>>>(d_A, d_B, d_C, m, n);

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(stop));


    // Copy the device result matrix in device memory
    // to the host result matrix in host memory
    printf("Copy output data from the CUDA device to the host memory\n");
    CUDA_CHECK(cudaMemcpy(h_C, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify that the result matrix is correct
    common_verify_matAdd(h_A, h_B, h_C, m, n);
    
    // Compute and Print the performance
    COMPUTE_MATADD_PERFORMANCE(start, stop, m, n, threads.x * threads.y);
    
    // Free device global memory
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    printf("Done\n");

    return 0;
}

void Usage(char prog_name[])
{
    fprintf(stderr, "Usage: %s <m> <n>\n", prog_name);
    fprintf(stderr, "\t<m> : the number of rows in matrix A, B.\n");
    fprintf(stderr, "\t<n> : the number of columns in matrix A, B.\n");
    exit(EXIT_FAILURE);
}

__global__
void matrixAdd(const float *A, const float *B, float *C, const int M, const int N)
{
    int ROW = blockIdx.x * blockDim.x + threadIdx.x;
    int COL = blockIdx.y * blockDim.y + threadIdx.y;

    if (ROW < M && COL < N) {
        C[(ROW * N) + COL] = A[(ROW * N) + COL] + B[(ROW * N) + COL];
    }
}