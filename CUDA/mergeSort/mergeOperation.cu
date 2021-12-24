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
 *      "--blocks=<N>"      : Specify the number of blocks (default: 32);
 *      "--tile=<N>"        : Specify the number of tile size (default: 1024)
 *****************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <random>
#include <algorithm>
#include <cassert>
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
__global__ void merge_tiled_kernel(int* A, int m, int* B, int n, int* C, int tile_size);

__host__ __device__ int co_rank_circular(int k, int* A, int m, int* B, int n, int A_S_start, int B_S_start, int tile_size);
__host__ __device__ void sequentialMergeCircular(int* A, int m, int* B, int n, int* C, int A_S_start, int B_S_start, int tile_size);
__global__ void merge_circular_buffer_kernel(int* A, int m, int* B, int n, int* C, int tile_size);

int main(int argc, char** argv)
{
    printf("[Parallel Merge Operation...]\n\n");
    int m = 1<<22;
    int n = 1<<22;
    int threads = 512;
    int blocks = 32;
    int tile_size = 1024;

    if (checkCmdLineFlag(argc, (const char **)argv, "m")) {
        m = getCmdLineArgumentInt(argc, (const char **)argv, "m");
    }
    if (checkCmdLineFlag(argc, (const char **)argv, "n")) {
        n = getCmdLineArgumentInt(argc, (const char **)argv, "n");
    }
    if (checkCmdLineFlag(argc, (const char **)argv, "threads")) {
        threads = getCmdLineArgumentInt(argc, (const char **)argv, "threads");
    }
    if (checkCmdLineFlag(argc, (const char **)argv, "blocks")) {
        blocks = getCmdLineArgumentInt(argc, (const char **)argv, "blocks");
    }
    if (checkCmdLineFlag(argc, (const char **)argv, "tile")) {
        tile_size = getCmdLineArgumentInt(argc, (const char **)argv, "tile");
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

    // tiled parallel merge operation
    int smem_size = tile_size*2*4; // bytes
    printf("\n[Tiled Parallel Merge Operation...]\n");
    printf("The number of threads per block: %d\n", threads);
    printf("The number of blocks in Grid: %d\n", blocks);
    printf("The number of tiles: %d\n", tile_size);
    printf("The size of shared memory per block: %d bytes\n", smem_size);
    GET_TIME(start);
    merge_tiled_kernel<<<blocks, threads, smem_size>>>(d_in, m, d_in+m, n, d_out, tile_size);
    CUDA_CHECK(cudaMemcpy(h_out, d_out, (m+n)*sizeof(int), cudaMemcpyDeviceToHost));
    GET_TIME(finish);
    printf("\tElapsed Time: %.6f msec\n", (finish-start)*1000);
    printf(checkValid(h_out, n) ? "PASSED\n" : "FAILED\n");

#ifdef DEBUG
    printArray(h_out, m+n);
    printf("\n");
#endif

    // circular-buffer version of tiled parallel merge operation
    smem_size = tile_size*2*4; // bytes
    printf("\n[Tiled Parallel Merge Operation With Circular-Buffer...]\n");
    printf("The number of threads per block: %d\n", threads);
    printf("The number of blocks in Grid: %d\n", blocks);
    printf("The number of tiles: %d\n", tile_size);
    printf("The size of shared memory per block: %d bytes\n", smem_size);
    GET_TIME(start);
    merge_circular_buffer_kernel<<<blocks, threads, smem_size>>>(d_in, m, d_in+m, n, d_out, tile_size);
    CUDA_CHECK(cudaMemcpy(h_out, d_out, (m+n)*sizeof(int), cudaMemcpyDeviceToHost));
    GET_TIME(finish);
    printf("\tElapsed Time: %.6f msec\n", (finish-start)*1000);
    printf(checkValid(h_out, n) ? "PASSED\n" : "FAILED\n");

#ifdef DEBUG
    printArray(h_out, m+n);
    printf("\n");
#endif

    free(h_in);
    free(h_out);
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));

    return 0;
}

bool checkValid(int* arr, int size)
{
    for (int i = 0; i < size; i++) {
        if (arr[i] != i) {
            printf("failed at %d = %d\n", i, arr[i]);
            return false;
        }
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

__global__
void merge_tiled_kernel(int* A, int m, int* B, int n, int* C, int tile_size)
{
    /* Part 1 : Identifying block-level output and input subarrays */
    extern __shared__ int shareAB[];
    int* A_S = shareAB;
    int* B_S = shareAB + tile_size;
    int C_curr = blockIdx.x * ceil((m+n)/(float)gridDim.x);
    int C_next = min((blockIdx.x+1) * (int)ceil((m+n)/(float)gridDim.x), m+n);

    if (threadIdx.x == 0) {
        A_S[0] = co_rank(C_curr, A, m, B, n);
        A_S[1] = co_rank(C_next, A, m, B, n);
    }
    __syncthreads();

    int A_curr = A_S[0];
    int A_next = A_S[1];
    int B_curr = C_curr - A_curr;
    int B_next = C_next - A_next;
    __syncthreads();

    int counter = 0;
    int C_length = C_next - C_curr;
    int A_length = A_next - A_curr;
    int B_length = B_next - B_curr;
    int total_iteration = ceil((C_length)/(float)tile_size);
    int C_completed = 0;
    int A_consumed = 0;
    int B_consumed = 0;

    while (counter < total_iteration) {
        /* Part 2 : Loading A and B elements into the shared memory */
        // loading tile-size A and B elements into shared memory
        for (int i = 0; i < tile_size; i += blockDim.x) {
            if (i + threadIdx.x < A_length - A_consumed)
                A_S[i + threadIdx.x] = A[A_curr + A_consumed + i + threadIdx.x];
        }
        for (int i = 0; i < tile_size; i += blockDim.x) {
            if (i + threadIdx.x < B_length - B_consumed)
                B_S[i + threadIdx.x] = B[B_curr + B_consumed + i + threadIdx.x];
        }
        __syncthreads();
        
        /* Part 3 : All threads merge their individual subarrays in parallel */
        int c_curr = threadIdx.x * (tile_size/blockDim.x);
        int c_next = (threadIdx.x+1) * (tile_size/blockDim.x);
        c_curr = (c_curr <= C_length - C_completed) ? c_curr : C_length - C_completed;
        c_next = (c_next <= C_length - C_completed) ? c_next : C_length - C_completed;
        
        // find co-rank for c_curr and c_next
        int a_curr = co_rank(c_curr, A_S, min(tile_size, A_length-A_consumed), B_S, min(tile_size, B_length-B_consumed));
        int b_curr = c_curr - a_curr;
        int a_next = co_rank(c_next, A_S, min(tile_size, A_length-A_consumed), B_S, min(tile_size, B_length-B_consumed));
        int b_next = c_next - a_next;

        // All threads call the sequential merge function
        sequentialMerge(A_S + a_curr, a_next - a_curr, 
                    B_S + b_curr, b_next - b_curr, 
                    C + C_curr + C_completed + c_curr);
        // Update the A and B elements that have been consumed thus far
        counter++;
        C_completed += tile_size;
        A_consumed += co_rank(tile_size, A_S, tile_size, B_S, tile_size);
        B_consumed = C_completed - A_consumed;
        __syncthreads();
    }
}

__host__ __device__
int co_rank_circular(int k, int* A, int m, int* B, int n, int A_S_start, int B_S_start, int tile_size)
{
    int i = (k < m) ? k : m; // i = min(k, m);
    int j = k-i;
    int i_low = (0 > (k-n)) ? 0 : k-n; // i_low = max(0, k-n);
    int j_low = (0 > (k-m)) ? 0 : k-m; // j_low = max(0, k-m);
    int delta;
    bool active = true;

    while (active) {
        int i_cir = (A_S_start + i >= tile_size) ? A_S_start + i - tile_size : A_S_start + i;
        int i_m_1_cir = (A_S_start + i - 1 >= tile_size) ? A_S_start + i - 1 - tile_size : A_S_start + i - 1;
        int j_cir = (B_S_start + j >= tile_size) ? B_S_start + j - tile_size : B_S_start + j;
        int j_m_1_cir = (B_S_start + j - 1 >= tile_size) ? B_S_start + j - 1 - tile_size : B_S_start + j - 1;

        if (i > 0 && j < n && A[i_m_1_cir] > B[j_cir]) {
            delta = ((i - i_low + 1) >> 1);
            j_low = j;
            j = j + delta;
            i = i - delta;
        }
        else if (j > 0 && i < m && B[j_m_1_cir] >= A[i_cir]) {
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

__host__ __device__
void sequentialMergeCircular(int* A, int m, int* B, int n, int* C, int A_S_start, int B_S_start, int tile_size)
{
    int iA = 0, iB = 0, iC = 0;
    while ((iA < m) && (iB < n)) {
        int iA_cir = (A_S_start + iA >= tile_size) ? A_S_start + iA - tile_size : A_S_start + iA;
        int iB_cir = (B_S_start + iB >= tile_size) ? B_S_start + iB - tile_size : B_S_start + iB;
        if (A[iA_cir] <= B[iB_cir]) {
            C[iC++] = A[iA_cir];
            iA++;
        }
        else {
            C[iC++] = B[iB_cir];
            iB++;
        }
    }

    if (iA == m) {
        while (iB < n) {
            int iB_cir = (B_S_start + iB >= tile_size) ? B_S_start + iB - tile_size : B_S_start + iB;
            C[iC++] = B[iB_cir];
            iB++;
        }
    }
    else {
        while (iA < m) {
            int iA_cir = (A_S_start + iA >= tile_size) ? A_S_start + iA - tile_size : A_S_start + iA;
            C[iC++] = A[iA_cir];
            iA++;
        }
    }
}

__global__
void merge_circular_buffer_kernel(int* A, int m, int* B, int n, int* C, int tile_size)
{
    /* Part 1 : Identifying block-level output and input subarrays */
    extern __shared__ int shareAB[];
    int* A_S = shareAB;
    int* B_S = shareAB + tile_size;
    int C_curr = blockIdx.x * ceil((m+n)/(float)gridDim.x);
    int C_next = min((blockIdx.x+1) * (int)ceil((m+n)/(float)gridDim.x), m+n);

    if (threadIdx.x == 0) {
        A_S[0] = co_rank(C_curr, A, m, B, n);
        A_S[1] = co_rank(C_next, A, m, B, n);
    }
    __syncthreads();

    int A_curr = A_S[0];
    int A_next = A_S[1];
    int B_curr = C_curr - A_curr;
    int B_next = C_next - A_next;
    __syncthreads();

    int counter = 0;
    int C_length = C_next - C_curr;
    int A_length = A_next - A_curr;
    int B_length = B_next - B_curr;
    int total_iteration = ceil((C_length)/(float)tile_size);
    int C_completed = 0;
    int A_consumed = 0;
    int B_consumed = 0;

    int A_S_start = 0;
    int B_S_start = 0;
    int A_S_consumed = tile_size; // in the first iteration, fill the tile_size
    int B_S_consumed = tile_size; // in the first iteration, fill the tile_size

    while (counter < total_iteration) {
        /* Part 2 : Loading A and B elements into the shared memory */
        // loading A_S_consumed elements into A_S
        for (int i = 0; i < A_S_consumed; i += blockDim.x) {
            if (i + threadIdx.x < A_length - A_consumed && i + threadIdx.x < A_S_consumed)
                A_S[(A_S_start + i + threadIdx.x) % tile_size] = A[A_curr + A_consumed + i + threadIdx.x];
        }
        // loading B_S_consumed elements into B_S
        for (int i = 0; i < B_S_consumed; i += blockDim.x) {
            if (i + threadIdx.x < B_length - B_consumed && i + threadIdx.x < B_S_consumed)
                B_S[(B_S_start + i + threadIdx.x) % tile_size] = B[B_curr + B_consumed + i + threadIdx.x];
        }
        __syncthreads();
        
        /* Part 3 : All threads merge their individual subarrays in parallel */
        int c_curr = threadIdx.x * (tile_size/blockDim.x);
        int c_next = (threadIdx.x+1) * (tile_size/blockDim.x);
        c_curr = (c_curr <= C_length - C_completed) ? c_curr : C_length - C_completed;
        c_next = (c_next <= C_length - C_completed) ? c_next : C_length - C_completed;
        
        // find co-rank for c_curr and c_next
        int a_curr = co_rank_circular(c_curr,
                                    A_S, min(tile_size, A_length-A_consumed),
                                    B_S, min(tile_size, B_length-B_consumed),
                                    A_S_start, B_S_start, tile_size);
        int b_curr = c_curr - a_curr;
        int a_next = co_rank_circular(c_next,
                                    A_S, min(tile_size, A_length-A_consumed),
                                    B_S, min(tile_size, B_length-B_consumed),
                                    A_S_start, B_S_start, tile_size);
        int b_next = c_next - a_next;

        // All threads call the circular-buffer version of sequential merge function
        sequentialMergeCircular(A_S, a_next - a_curr, 
                            B_S, b_next - b_curr, 
                            C + C_curr + C_completed + c_curr,
                            A_S_start + a_curr, B_S_start + b_curr, tile_size);
        // Update the A and B elements that have been consumed thus far
        counter++;
        A_S_consumed = co_rank_circular(min(tile_size, C_length-C_completed),
                                        A_S, min(tile_size, A_length-A_consumed),
                                        B_S, min(tile_size, B_length-B_consumed),
                                        A_S_start, B_S_start, tile_size);
        B_S_consumed = min(tile_size, C_length-C_completed) - A_S_consumed;
        A_consumed += A_S_consumed;
        C_completed += min(tile_size, C_length-C_completed);
        B_consumed = C_completed - A_consumed;

        A_S_start = A_S_start + A_S_consumed;
        if (A_S_start >= tile_size)
            A_S_start = A_S_start - tile_size;
        B_S_start = B_S_start + B_S_consumed;
        if (B_S_start >= tile_size)
            B_S_start = B_S_start - tile_size; 

        __syncthreads();
    }
}