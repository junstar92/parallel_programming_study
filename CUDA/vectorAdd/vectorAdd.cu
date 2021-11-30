/*****************************************************************************
 * File:        vectorAdd.cu
 * Description: Vector addition, C = A + B
 *              This code is a basic sample that implements element by element
 *              vector addition.
 *              
 * Compile:     nvcc -o vectorAdd vectorAdd.cu -I.. -lcuda
 * Run:         ./vectorAdd <n>
 *                  <n> : the number of elements in vector
 *****************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <common/common.h>

void Usage(char prog_name[]);
__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements);

int main(int argc, char* argv[])
{
    if (argc != 2) {
        Usage(argv[0]);
    }

    int numElements = strtol(argv[1], NULL, 10);
    printf("[Vector addition of %d elements]\n", numElements);

    // Allocate the host input vectors A, B, C
    float *h_A = (float*)malloc(numElements * sizeof(float));
    float *h_B = (float*)malloc(numElements * sizeof(float));
    float *h_C = (float*)malloc(numElements * sizeof(float));

    // Verify that allocations succeeded
    if (h_A == NULL || h_B == NULL || h_C == NULL) {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_SUCCESS);
    }

    // Initialize that host input vectors
    common_init_rand_fvec(h_A, numElements);
    common_init_rand_fvec(h_B, numElements);

    // Allocate the device input vectors A, B, C
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void**)&d_A, numElements * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_B, numElements * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_C, numElements * sizeof(float)));

    // Copy the host input vector A and B in host memory 
    // to the device input vectors in device memory
    printf("Copy input data from the host memory to the CUDA device\n");
    CUDA_CHECK(cudaMemcpy(d_A, h_A, numElements * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, numElements * sizeof(float), cudaMemcpyHostToDevice));

    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    double start, finish;
    GET_TIME(start);
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
    CUDA_CHECK(cudaDeviceSynchronize());
    GET_TIME(finish);
    CUDA_CHECK(cudaGetLastError());

    // Copy the device result vector in device memory
    // to the host result vector in host memory
    printf("Copy output data from the CUDA device to the host memory\n");
    CUDA_CHECK(cudaMemcpy(h_C, d_C, numElements * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify that the result vector is correct
    printf("Random Sampling Verifying...\n");
    for (int i = 0; i < 10; i++) {
        int idx = rand() % numElements;
        printf("[INDEX %d] %f + %f = %f\n", idx, h_A[idx], h_B[idx], h_C[idx]);
        if (fabs(h_A[idx] + h_B[idx] - h_C[idx]) > 1e-5) {
            fprintf(stderr, "Result verification failed at element %d\n", idx);
            exit(EXIT_FAILURE);
        }
    }
    printf(".....\n");
    printf("Test PASSED\n");
    
    // Free device global memory
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    printf("Done\n");
    printf("Elapsed time of kernel function : %f seconds\n", finish-start);
    return 0;
}

void Usage(char prog_name[])
{
    fprintf(stderr, "Usage: %s <n>\n", prog_name);
    fprintf(stderr, "\t<n> : the number of elements in vector\n");
    exit(EXIT_FAILURE);
}

__global__
void vectorAdd(const float *A, const float *B, float *C, int numElements)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < numElements)
        C[i] = A[i] + B[i];
}