/*****************************************************************************
 * File:        mergeOperation.cu
 * Description: Implement sequential and parallel Merge Operation.
 *              
 *              
 * Compile:     nvcc -o mergeOperation mergeOperation.cu -I..
 * Run:         ./mergeOperation
 * Argument:
 *      "--m=<N>"           : Specify the number of elements in array A (default: 1<<22)
 *      "--n=<N>"           : Specify the number of elements in array B (default: 1<<22)
 *      "--threads=<N>"     : Specify the number of threads per block (default: 512)
 *****************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <random>
#include <algorithm>
#include <cuda_runtime.h>

#include <common/common.h>
#include <common/common_string.h>

#ifdef DEBUG
void printArray(int* arr, int size)
{
    for (int i = 0; i < size; i++)
        printf("%d ", arr[i]);
    printf("\n");
}
#endif
bool checkValid(int* arr, int size);
__host__ __device__ void sequentialMerge(int* A, int m, int* B, int n, int* C);
__host__ __device__ int co_rank(int k, int* A, int m, int* B, int n);
__global__ void merge_basic_kernel(int* A, int m, int* B, int n, int* C);


int main(int argc, char** argv)
{
    printf("[Parallel Merge Operation...]\n\n");
    int m = 1<<22;
    int n = 1<<22;
    int threads = 512;

    if (checkCmdLineFlag(argc, (const char **)argv, "m")) {
        m = getCmdLineArgumentInt(argc, (const char **)argv, "m");
    }
    if (checkCmdLineFlag(argc, (const char **)argv, "n")) {
        n = getCmdLineArgumentInt(argc, (const char **)argv, "n");
    }
    if (checkCmdLineFlag(argc, (const char **)argv, "threads")) {
        threads = getCmdLineArgumentInt(argc, (const char **)argv, "threads");
    }

    printf("Size of input array A: %d\n", m);
    printf("Size of input array B: %d\n", n);

    int *h_in, *h_out;
    h_in = (int*)malloc((m+n)*sizeof(int));
    h_out = (int*)malloc((m+n)*sizeof(int));

    for (int i = 0; i < m+n; i++)
        h_in[i] = i;
    for (int i = 0; i < m+n; i++) {
        // Shuffle
        int r = rand() % (m+n);
        int tmp = h_in[i];
        h_in[i] = h_in[r];
        h_in[r] = tmp;
    }

    // sorting array A and B
    std::sort(h_in, h_in+m);
    std::sort(h_in+m, h_in+m+n);
#ifdef DEBUG
    printf("Array A:\n");
    printArray(h_in, m);
    printf("Array B:\n");
    printArray(h_in+m, n);
#endif

    double start, finish;
    // sequential merge operation
    printf("\n[Sequential Merge Operation...]\n");
    GET_TIME(start);
    sequentialMerge(h_in, m, h_in+m, n, h_out);
    GET_TIME(finish);
    printf("\tElapsed Time: %.6f msec\n", (finish-start)*1000);
    printf(checkValid(h_out, m+n) ? "PASSED\n" : "FAILED\n");

#ifdef DEBUG
    printArray(h_out, m+n);
    printf("\n");
#endif

    // allocate device memory
    int *d_in, *d_out;
    CUDA_CHECK(cudaMalloc((void**)&d_in, (m+n)*sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_out, (m+n)*sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_in, h_in, (m+n)*sizeof(int), cudaMemcpyHostToDevice));

    // basic parallel merge operation
    printf("\n[Basic Parallel Merge Operation...]\n");
    int blocks = (m+n + threads - 1) / threads;
    printf("The number of threads per block: %d\n", threads);
    printf("The number of blocks in Grid: %d\n", blocks);
    GET_TIME(start);
    merge_basic_kernel<<<blocks, threads>>>(d_in, m, d_in+m, n, d_out);
    CUDA_CHECK(cudaMemcpy(h_out, d_out, (m+n)*sizeof(int), cudaMemcpyDeviceToHost));
    GET_TIME(finish);
    printf("\tElapsed Time: %.6f msec\n", (finish-start)*1000);
    printf(checkValid(h_out, n) ? "PASSED\n" : "FAILED\n");

#ifdef DEBUG
    printArray(h_out, m+n);
    printf("\n");
#endif

    int A[5] = {1,7,8,9,10};
    int B[4] = {7,10,10,12};

    merge_basic_kernel<<<1, 2>>>(A, 5, B, 4, h_out);

    free(h_in);
    free(h_out);
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));

    return 0;
}

bool checkValid(int* arr, int size)
{
    for (int i = 0; i < size; i++) {
        if (arr[i] != i)
            return false;
    }

    return true;
}

__host__ __device__
void sequentialMerge(int* A, int m, int* B, int n, int* C)
{
    int iA = 0, iB = 0, iC = 0;
    while ((iA < m) && (iB < n)) {
        if (A[iA] <= B[iB]) {
            C[iC++] = A[iA++];
        }
        else {
            C[iC++] = B[iB++];
        }
    }

    if (iA == m) {
        while (iB < n)
            C[iC++] = B[iB++];
    }
    else {
        while (iA < m)
            C[iC++] = A[iA++];
    }
}

__host__ __device__
int co_rank(int k, int* A, int m, int* B, int n)
{
    int i = (k < m) ? k : m; // i = min(k, m);
    int j = k-i;
    int i_low = (0 > (k-n)) ? 0 : k-n; // i_low = max(0, k-n);
    int j_low = (0 > (k-m)) ? 0 : k-m; // j_low = max(0, k-m);
    int delta;
    bool active = true;

    while (active) {
        if (i > 0 && j < n && A[i-1] > B[j]) {
            delta = ((i - i_low + 1) >> 1);
            j_low = j;
            j = j + delta;
            i = i - delta;
        }
        else if (j > 0 && i < m && B[j-1] >= A[i]) {
            delta = ((j - j_low + 1) >> 1);
            i_low = i;
            i = i + delta;
            j = j - delta;
        }
        else {
            active = false;
        }
    }

    return i;
}

__global__
void merge_basic_kernel(int* A, int m, int* B, int n, int* C)
{
    int tid = blockDim.x*blockIdx.x + threadIdx.x;
    int k_curr = tid * ceil((m+n)/(float)(blockDim.x*gridDim.x));
    int k_next = min((tid+1) * (int)ceil((m+n)/(float)(blockDim.x*gridDim.x)), m+n);
    int i_curr = co_rank(k_curr, A, m, B, n);
    int i_next = co_rank(k_next, A, m, B, n);
    int j_curr = k_curr - i_curr;
    int j_next = k_next - i_next;
    
    sequentialMerge(A+i_curr, i_next-i_curr, B+j_curr, j_next-j_curr, C+k_curr);
}