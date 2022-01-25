/*****************************************************************************
 * File:        floating-point-perf.cu
 * Description: This is an example for illustrating the performance cost of
 *              using double-precision floating-point values, relative to
 *              single-precision floating-point values.
 *              The core computational keernel executes a number of mathematical
 *              operations on an input value. This example includes a kernel for
 *              both single- and double-precision floating-point. Timing statements
 *              are placed around the copy of inputs to the device, the copy of
 *              outputs from the device, and the execution of the kernel.
 *              These timing statements enable comparison of overhead from both
 *              communication and computation.
 *              
 * Compile:     nvcc -o floating-point-perf floating-point-perf.cu -I..
 * Run:         ./floating-point-perf
 *****************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "common/common.h"

/* The computational kernel for single-precision floating-point */
__global__
void lots_of_float_compute(float* in, int N, size_t nIters, float* out)
{
    size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    size_t nThreads = gridDim.x * blockDim.x;

    for (; tid < N; tid += nThreads) {
        float val = in[tid];

        for (size_t i = 0; i < nIters; i++) {
            val = (val + 5.0f) - 101.0f;
            val = (val / 3.0f) + 102.0f;
            val = (val + 1.07f) - 103.0f;
            val = (val / 1.037f) + 104.0f;
            val = (val + 3.00f) - 105.0f;
            val = (val / 0.22f) + 106.0f;
        }

        out[tid] = val;
    }
}

/* The computational kernel for double-precision floating-point */
__global__
void lots_of_double_compute(double* in, int N, size_t nIters, double* out)
{
    size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    size_t nThreads = gridDim.x * blockDim.x;

    for (; tid < N; tid += nThreads) {
        double val = in[tid];

        for (size_t i = 0; i < nIters; i++) {
            val = (val + 5.0f) - 101.0f;
            val = (val / 3.0f) + 102.0f;
            val = (val + 1.07f) - 103.0f;
            val = (val / 1.037f) + 104.0f;
            val = (val + 3.00f) - 105.0f;
            val = (val / 0.22f) + 106.0f;
        }

        out[tid] = val;
    }
}

static void run_float_test(size_t N, int nIters, int blocksPerGrid,
                        int threadsPerBlock, double *toDeviceTime,
                        double *kernelTime, double *fromDeviceTime,
                        float *sample, int sampleLength)
{
    float *h_floatInputs, *h_floatOutputs;
    float *d_floatInputs, *d_floatOutputs;

    h_floatInputs = (float*)malloc(sizeof(float)*N);
    h_floatOutputs = (float*)malloc(sizeof(float)*N);
    CUDA_CHECK(cudaMalloc((void**)&d_floatInputs, sizeof(float) * N));
    CUDA_CHECK(cudaMalloc((void**)&d_floatOutputs, sizeof(float) * N));

    for (int i = 0; i < N; i++)
        h_floatInputs[i] = (float)i;

    double start, stop;
    GET_TIME(start);
    CUDA_CHECK(cudaMemcpy(d_floatInputs, h_floatInputs, sizeof(float)*N, cudaMemcpyHostToDevice));
    GET_TIME(stop);
    *toDeviceTime = stop - start;

    GET_TIME(start);
    lots_of_float_compute<<<blocksPerGrid, threadsPerBlock>>>(d_floatInputs, N, nIters, d_floatOutputs);
    CUDA_CHECK(cudaDeviceSynchronize());
    GET_TIME(stop);
    *kernelTime = stop - start;

    GET_TIME(start);
    CUDA_CHECK(cudaMemcpy(h_floatOutputs, d_floatOutputs, sizeof(float)*N, cudaMemcpyDeviceToHost));
    GET_TIME(stop);
    *fromDeviceTime = stop - start;

    for (int i = 0; i < sampleLength; i++)
        sample[i] = h_floatOutputs[i];

    CUDA_CHECK(cudaFree(d_floatInputs));
    CUDA_CHECK(cudaFree(d_floatOutputs));
    free(h_floatInputs);
    free(h_floatOutputs);
}

static void run_double_test(size_t N, int nIters, int blocksPerGrid,
                        int threadsPerBlock, double *toDeviceTime,
                        double *kernelTime, double *fromDeviceTime,
                        double *sample, int sampleLength)
{
    double *h_doubleInputs, *h_doubleOutputs;
    double *d_doubleInputs, *d_doubleOutputs;

    h_doubleInputs = (double*)malloc(sizeof(double)*N);
    h_doubleOutputs = (double*)malloc(sizeof(double)*N);
    CUDA_CHECK(cudaMalloc((void**)&d_doubleInputs, sizeof(double) * N));
    CUDA_CHECK(cudaMalloc((void**)&d_doubleOutputs, sizeof(double) * N));

    for (int i = 0; i < N; i++)
        h_doubleInputs[i] = (double)i;

    double start, stop;
    GET_TIME(start);
    CUDA_CHECK(cudaMemcpy(d_doubleInputs, h_doubleInputs, sizeof(double)*N, cudaMemcpyHostToDevice));
    GET_TIME(stop);
    *toDeviceTime = stop - start;

    GET_TIME(start);
    lots_of_double_compute<<<blocksPerGrid, threadsPerBlock>>>(d_doubleInputs, N, nIters, d_doubleOutputs);
    CUDA_CHECK(cudaDeviceSynchronize());
    GET_TIME(stop);
    *kernelTime = stop - start;

    GET_TIME(start);
    CUDA_CHECK(cudaMemcpy(h_doubleOutputs, d_doubleOutputs, sizeof(double)*N, cudaMemcpyDeviceToHost));
    GET_TIME(stop);
    *fromDeviceTime = stop - start;

    for (int i = 0; i < sampleLength; i++)
        sample[i] = h_doubleOutputs[i];

    CUDA_CHECK(cudaFree(d_doubleInputs));
    CUDA_CHECK(cudaFree(d_doubleOutputs));
    free(h_doubleInputs);
    free(h_doubleOutputs);
}

int main(int argc, char** argv)
{

    cudaDeviceProp deviceProp;
    size_t totalMem, freeMem;

    CUDA_CHECK(cudaMemGetInfo(&freeMem, &totalMem));
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, 0));

    size_t N = (freeMem * 0.9 / 2) / sizeof(double);
    int threadsPerBlock = 255;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    if (blocksPerGrid > deviceProp.maxGridSize[0]) {
        blocksPerGrid = deviceProp.maxGridSize[0];
    }

    printf("Running %d blocks with %d threads/block over %lu elements\n", blocksPerGrid, threadsPerBlock, N);

    int sampleLength = 10;
    int nRuns = 5;
    int nKernelIters = 20;
    
    float *floatSample = (float*)malloc(N * sizeof(float));
    double *doubleSample = (double*)malloc(N * sizeof(double));

    double meanFloatToDeviceTime{0.0}, meanFloatKernelTime{0.0}, meanFloatFromDeviceTime{0.0};
    double meanDoubleToDeviceTime{0.0}, meanDoubleKernelTime{0.0}, meanDoubleFromDeviceTime{0.0};

    for (int i = 0; i < nRuns; i++) {
        double toDeviceTime, kernelTime, fromDeviceTime;

        run_float_test(N, nKernelIters, blocksPerGrid, threadsPerBlock,
                        &toDeviceTime, &kernelTime, &fromDeviceTime,
                        floatSample, sampleLength);
        meanFloatToDeviceTime += toDeviceTime;
        meanFloatKernelTime += kernelTime;
        meanFloatFromDeviceTime += fromDeviceTime;

        run_double_test(N, nKernelIters, blocksPerGrid, threadsPerBlock,
                        &toDeviceTime, &kernelTime, &fromDeviceTime,
                        doubleSample, sampleLength);
        meanDoubleToDeviceTime += toDeviceTime;
        meanDoubleKernelTime += kernelTime;
        meanDoubleFromDeviceTime += fromDeviceTime;

        if (i == 0) {
            printf("\nInput\tDiff Between Single- and Double-Precision\n");
            printf("------\t------\n");

            for (int j = 0; j < sampleLength; j++) {
                printf("%d\t%.20e\n", j, fabs(doubleSample[j] - static_cast<double>(floatSample[j])));
            }
            printf("\n");
        }
    }
    
    meanFloatToDeviceTime /= nRuns;
    meanFloatKernelTime /= nRuns;
    meanFloatFromDeviceTime /= nRuns;
    meanDoubleToDeviceTime /= nRuns;
    meanDoubleKernelTime /= nRuns;
    meanDoubleFromDeviceTime /= nRuns;

    printf("For single-precision floating point, mean times for:\n");
    printf("  Copy to device:   %f s\n", meanFloatToDeviceTime);
    printf("  Kernel execution: %f s\n", meanFloatKernelTime);
    printf("  Copy from device: %f s\n", meanFloatFromDeviceTime);
    printf("For double-precision floating point, mean times for:\n");
    printf("  Copy to device:   %f s (%.2fx slower than single-precision)\n",
           meanDoubleToDeviceTime,
           meanDoubleToDeviceTime / meanFloatToDeviceTime);
    printf("  Kernel execution: %f s (%.2fx slower than single-precision)\n",
           meanDoubleKernelTime,
           meanDoubleKernelTime / meanFloatKernelTime);
    printf("  Copy from device: %f s (%.2fx slower than single-precision)\n",
           meanDoubleFromDeviceTime,
           meanDoubleFromDeviceTime / meanFloatFromDeviceTime);

    return 0;
}