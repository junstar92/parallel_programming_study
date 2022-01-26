/*****************************************************************************
 * File:        atomic-ordering.cu
 * Description: This is an example to illustrates the difference between using
 *              atomic operations and using unsafe accesses to increment a shared
 *              variable.
 *              In both the atomics() and unsafe() kernels, each thread repeatedly
 *              increments a globally shared variable by 1. Each thread also stores
 *              the value is reads from the shared location for the first increment.
 *              
 * Compile:     nvcc -o atomic-ordering atomic-ordering.cu -I..
 * Run:         ./atomic-ordering
 *****************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "common/common.h"

__global__
void atomics(int* shared_var, int* values_read, int N, int iters)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid > N)
        return;

    values_read[tid] = atomicAdd(shared_var, 1);
    
    for (int i = 0; i < iters; i++)
        atomicAdd(shared_var, 1);
}

__global__
void unsafe(int* shared_var, int* values_read, int N, int iters)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid > N)
        return;

    int old = *shared_var;
    *shared_var = old + 1;
    values_read[tid] = old;

    for (int i = 0; i < iters; i++) {
        int old = *shared_var;
        *shared_var = old + 1;
    }
}

void print_read_results(int *h_arr, int *d_arr, int N, const char* label)
{
    int maxNumToPrint = 10;
    int nToPrint = N > maxNumToPrint ? maxNumToPrint : N;

    CUDA_CHECK(cudaMemcpy(h_arr, d_arr, nToPrint * sizeof(int), cudaMemcpyDeviceToHost));
    printf("Threads performing %s operations read values", label);

    for (int i = 0; i < nToPrint; i++) {
        printf(" %d", h_arr[i]);
    }
    printf("\n");
}

int main(int argc, char** argv)
{
    int N = 64;
    int block = 32;
    int runs = 30;
    int iters = 100000;
    int *d_shared_var;
    int h_shared_var_atomic, h_shared_var_unsafe;
    int *d_values_read_atomic, *d_values_read_unsafe;
    int *h_values_read;

    CUDA_CHECK(cudaMalloc((void**)&d_shared_var, sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_values_read_atomic, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_values_read_unsafe, N * sizeof(int)));
    h_values_read = (int*)malloc(N * sizeof(int));

    double atomic_mean_time = 0;
    double unsafe_mean_time = 0;

    for (int r = 0; r < runs; r++) {
        double start, stop;
        GET_TIME(start);
        CUDA_CHECK(cudaMemset(d_shared_var, 0x00, sizeof(int)));
        atomics<<<N / block, block>>>(d_shared_var, d_values_read_atomic, N, iters);
        CUDA_CHECK(cudaDeviceSynchronize());
        GET_TIME(stop)
        atomic_mean_time += (stop - start);
        CUDA_CHECK(cudaMemcpy(&h_shared_var_atomic, d_shared_var, sizeof(int), cudaMemcpyDeviceToHost));

        GET_TIME(start);
        CUDA_CHECK(cudaMemset(d_shared_var, 0x00, sizeof(int)));
        unsafe<<<N / block, block>>>(d_shared_var, d_values_read_unsafe, N, iters);
        CUDA_CHECK(cudaDeviceSynchronize());
        GET_TIME(stop);
        unsafe_mean_time += stop - start;
        CUDA_CHECK(cudaMemcpy(&h_shared_var_unsafe, d_shared_var, sizeof(int), cudaMemcpyDeviceToHost));
    }

    printf("In total, %d runs using atomic operations took %f s\n", runs, atomic_mean_time);
    printf("  Using atomic operations also produced an output of %d\n", h_shared_var_atomic);
    printf("In total, %d runs using unsafe operations took %f s\n", runs, unsafe_mean_time);
    printf("  Using unsafe operations also produced an output of %d\n", h_shared_var_unsafe);

    print_read_results(h_values_read, d_values_read_atomic, N, "atomic");
    print_read_results(h_values_read, d_values_read_unsafe, N, "unsafe");

    return 0;
}