/*****************************************************************************
 * File:        intrinsic-standard-comp.cu
 * Description: This is an example to demonstrate the relative performance and
 *              accuracy of CUDA standard and intrinsic functions.
 * 
 *              The computational kernel of this example is the iterative 
 *              calculation of a value squared. This computation is done on the
 *              host, on the device with a standard function. The results from
 *              all three are compared for numerical accuarcy (with the host as
 *              the baseline), and the performance of standard and intrinsic 
 *              function is also compared.
 *              
 * Compile:     nvcc -o intrinsic-standard-comp intrinsic-standard-comp.cu -I..
 * Run:         ./intrinsic-standard-comp
 *****************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "common/common.h"

/* Perform iters power operations using the standard powf function. */
__global__
void standard_kernel(float a, float *out, int iters)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid == 0) {
        float tmp;

        for (int i = 0; i < iters; i++)
            tmp = powf(a, 2.0f);
        
        *out = tmp;
    }
}

/* Perform iters power operations using the intrinsic __powf function. */
__global__
void intrinsic_kernel(float a, float *out, int iters)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid == 0) {
        float tmp;

        for (int i = 0; i < iters; i++)
            tmp = __powf(a, 2.0f);
        
        *out = tmp;
    }
}

int main(int argc, char** argv)
{
    int runs = 30;
    int iters = 1000;

    float *d_standard_out, h_standard_out;
    CUDA_CHECK(cudaMalloc((void**)&d_standard_out, sizeof(float)));

    float *d_intrinsic_out, h_intrinsic_out;
    CUDA_CHECK(cudaMalloc((void**)&d_intrinsic_out, sizeof(float)));

    float input_value = 8181.25;

    double mean_standard_time = 0.0;
    double mean_intrinsic_time = 0.0;

    for (int i = 0; i < runs; i++) {
        double start, stop;

        GET_TIME(start);
        standard_kernel<<<1, 32>>>(input_value, d_standard_out, iters);
        CUDA_CHECK(cudaDeviceSynchronize());
        GET_TIME(stop);
        mean_standard_time += stop - start;
        
        GET_TIME(start);
        intrinsic_kernel<<<1, 32>>>(input_value, d_intrinsic_out, iters);
        CUDA_CHECK(cudaDeviceSynchronize());
        GET_TIME(stop);
        mean_intrinsic_time += stop - start;
    }

    CUDA_CHECK(cudaMemcpy(&h_standard_out, d_standard_out, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_intrinsic_out, d_intrinsic_out, sizeof(float), cudaMemcpyDeviceToHost));
    float host_value = powf(input_value, 2.0f);

    mean_standard_time /= runs;
    mean_intrinsic_time /= runs;

    printf("Host calculated\t\t\t%f\n", host_value);
    printf("Standard Device calculated\t%f\n", h_standard_out);
    printf("Intrinsic Device calculated\t%f\n", h_intrinsic_out);
    printf("Host equals Standard?\t\t%s, diff=%e\n",
           host_value == h_standard_out ? "Yes" : "No",
           fabs(host_value - h_standard_out));
    printf("Host equals Intrinsic?\t\t%s, diff=%e\n",
           host_value == h_intrinsic_out ? "Yes" : "No",
           fabs(host_value - h_intrinsic_out));
    printf("Standard equals Intrinsic?\t%s, diff=%e\n",
           h_standard_out == h_intrinsic_out ? "Yes" : "No",
           fabs(h_standard_out - h_intrinsic_out));
    printf("\n");
    printf("Mean execution time for standard function powf:    %f ms\n",
           mean_standard_time * 1000.f);
    printf("Mean execution time for intrinsic function __powf: %f ms\n",
           mean_intrinsic_time * 1000.f);

    return 0;
}