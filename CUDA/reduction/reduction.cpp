/*****************************************************************************
 * File:        reduction.cpp
 * Description: Implement Sum Reduction with some reduction kernels
 *      [1]: simple sum reduction with highly divergent warps
 *      [2]: revised version from Kernel[1], there is a bank conflicts of shared memory
 *      [3]: Fix interleaved addressing problem in Kernel[2]. It is sequential addressing
 *      [4]: This version uses n/2 threads by revising Kernel[3].
 *           It performs the first level of reduction when reading from global memory.
 *      [5]: This version uses the warp suffle operation if available to reduce
 *           warp synchronization. When shuffle is not available the final warp's 
 *           worth of work is unrolled to reduce looping overhead
 *           See http://devblogs.nvidia.com/parallelforall/faster-parallel-reductions-kepler 
 *       ...
 *       TBD
 *              
 * Compile:     nvcc -o reduction reduction.cpp reductionKernel.cu -I.. -I. -lcuda
 * Run:         ./reduction
 * Argument:
 *      "--n=<N>"           : Specify the number of elements to reduce (default: 1 << 24)
 *      "--threads=<N>"     : Specify the number of threads per block (default: 256)
 *      "--maxblocks=<N>"   : Specify the maximum number of thread blocks (not applied)
 *      "--iteration=<N>"   : Specify the number of iteration (default: 100)
 *      "--kernel=<N>"      : Specify which kernel(sumReduce<N>) to run (default 1)
 *      "--type=<T>"        : The datatype forthe reduction. <T> is "int", "float"
 *                            or "double" (default: int)
 *****************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#include <common/common.h>
#include <common/common_string.h>
#include <reduction.h>

enum ReduceType {
    REDUCE_INT,
    REDUCE_FLOAT,
    REDUCE_DOUBLE
};

template <typename T>
bool run(int argc, char** argv, ReduceType dataType);
template <typename T>
T benchmarkReduce(int size, int numThreads, int numBlocks, int maxThreads, int maxBlocks, int smemSize, int whichKernel,
                int nIter, bool finalReduction, double& total_time, T* h_out, T* d_in, T* d_out);
const char* getReduceTypeString(const ReduceType type);
template <class T>
T reduceCPU(T *data, int size);
unsigned int nextPow2(unsigned int x);
void getNumBlocksAndThreads(int whichKernel, int n, int maxBlocks, int maxThreads, int &blocks, int &threads);

// Program main
int main(int argc, char** argv)
{
    printf("[Sum Reduction...]\n");

    char *typeInput = NULL;
    getCmdLineArgumentString(argc, (const char **)argv, "type", &typeInput);

    ReduceType dataType = REDUCE_INT;
    if (0 != typeInput) {
        if (!strcasecmp(typeInput, "float")) {
            dataType = REDUCE_FLOAT;
        }
        else if (!strcasecmp(typeInput, "double")) {
            dataType = REDUCE_DOUBLE;
        }
        else
        {
            printf("Type %s is not recognized. Using default type int.\n\n", typeInput);
        }
    }

    cudaDeviceProp devProp;
    int dev = 0;
    CUDA_CHECK(cudaSetDevice(dev));
    CUDA_CHECK(cudaGetDeviceProperties(&devProp, dev));
    printf("Using Device %d: %s\n", dev, devProp.name);
    printf("Reducing array of data type %s\n\n", getReduceTypeString(dataType));

    bool result = false;

    switch (dataType) {
        default:
        case REDUCE_INT:
            result = run<int>(argc, argv, dataType);
        break;
        
        case REDUCE_FLOAT:
            result = run<float>(argc, argv, dataType);
        break;
        
        case REDUCE_DOUBLE:
            result = run<double>(argc, argv, dataType);
        break;
    }

    printf(result ? "Test PASSED\n" : "Test FAILED!\n");

    return 0;
}

template <class T>
bool run(int argc, char** argv, ReduceType dataType)
{
    int size = 1 << 24;     // 16,777,216
    int maxThreads = 256;
    int whichKernel = 1;
    int maxBlocks = 64;
    bool finalReduce = true;
    int nIter = 100;

    if (checkCmdLineFlag(argc, (const char **)argv, "n")) {
        size = getCmdLineArgumentInt(argc, (const char **)argv, "n");
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "threads")) {
        maxThreads = getCmdLineArgumentInt(argc, (const char **)argv, "threads");
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "kernel")) {
        whichKernel = getCmdLineArgumentInt(argc, (const char **)argv, "kernel");
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "maxblocks")) {
        maxBlocks = getCmdLineArgumentInt(argc, (const char **)argv, "maxblocks");
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "iteration")) {
        nIter = getCmdLineArgumentInt(argc, (const char **)argv, "iteration");
    }

    printf("%d elements\n", size);
    printf("%d threads (max)\n", maxThreads);

    unsigned int bytes = size * sizeof(T);
    T* h_in = (T*)malloc(bytes);

    for (int i = 0; i < size; i++) {
        // Keep the numbers small so we don't get truncation error in the sum
        if (dataType == REDUCE_INT) {
            h_in[i] = (T)(rand() & 0xFF);
        } else {
            h_in[i] = (rand() & 0xFF) / (T)RAND_MAX;
        }
    }

    int numBlocks = 0;
    int numThreads = 0;
    getNumBlocksAndThreads(whichKernel, size, maxBlocks, maxThreads, numBlocks, numThreads);

    T* h_out = (T*)malloc(numBlocks * sizeof(T));
    //printf("%d threads\n", numThreads);
    printf("%d blocks\n", numBlocks);

    T* d_in, *d_out;
    CUDA_CHECK(cudaMalloc((void**)&d_in, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_out, numBlocks * sizeof(T)));

    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));
    
    // calculate shared memory per block
    int smemSize = (numThreads <= 32) ? 2 * numThreads * sizeof(T) : numThreads * sizeof(T);
    printf("Shared Memory Size per Block: %d bytes\n", smemSize);

    // warm up
    reduce<T>(size, numThreads, numBlocks, smemSize, whichKernel, d_in, d_out);

    double total_time = 0;
    T gpu_result = 0;

    gpu_result = benchmarkReduce<T>(size, numThreads, numBlocks, maxThreads, maxBlocks, smemSize,
                                    whichKernel, nIter, finalReduce, total_time, h_out, d_in, d_out);
    printf("The number of iteration: %d\n", nIter);
    double reduceTime = (total_time / (double)nIter); //sec
    printf("[Kernel %d] Throughput = %.4f GB/s, Time = %.5f ms, Size = %u Elements\n",
        whichKernel, ((double)bytes / reduceTime)*1.0e-9, reduceTime * 1000, size);

    T cpu_result = reduceCPU<T>(h_in, size);

    int precision = 0;
    double threshold = 0;
    double diff = 0;

    if (dataType == REDUCE_INT) {
        printf("\nGPU result = %d\n", (int)gpu_result);
        printf("CPU result = %d\n\n", (int)cpu_result);
    }
    else {
        if (dataType == REDUCE_FLOAT) {
            precision = 8;
            threshold = 1e-8 * size;
        }
        else {
            precision = 12;
            threshold = 1e-12 * size;
        }

        printf("\nGPU result = %.*f\n", precision, (double)gpu_result);
        printf("CPU result = %.*f\n\n", precision, (double)cpu_result);

        diff = fabs((double)gpu_result - (double)cpu_result);
    }

    // free memory
    free(h_in);
    free(h_out);
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));

    // validation check
    if (dataType == REDUCE_INT)
        return (gpu_result == cpu_result);
    else
        return (diff < threshold);
}

const char* getReduceTypeString(const ReduceType type)
{
    switch (type)
    {
    case REDUCE_INT:
        return "int";
    case REDUCE_FLOAT:
        return "float";
    case REDUCE_DOUBLE:
        return "double";
    default:
        return "unknown";
    }
}

template <typename T>
T benchmarkReduce(int size, int numThreads, int numBlocks, int maxThreads, int maxBlocks, int smemSize, 
                int whichKernel, int nIter, bool finalReduction, double& total_time, T* h_out, T* d_in, T* d_out)
{
    T gpu_result = 0;

    double start, finish;
    total_time = 0;
    for (int i = 0; i < nIter; i++) {
        gpu_result = 0;

        cudaDeviceSynchronize();
        GET_TIME(start);

        // execute the kernel
        reduce<T>(size, numThreads, numBlocks, smemSize, whichKernel, d_in, d_out);
        CUDA_CHECK(cudaGetLastError());

        if (finalReduction) {
            CUDA_CHECK(cudaMemcpy(h_out, d_out, numBlocks * sizeof(T), cudaMemcpyDeviceToHost));

            for (int i = 0; i < numBlocks; i++) {
                gpu_result += h_out[i];
            }
        }

        cudaDeviceSynchronize();
        GET_TIME(finish);
        total_time += (finish - start);
    }

    return gpu_result;
}

////////////////////////////////////////////////////////////////////////////////
//! Compute sum reduction on CPU
//! We use Kahan summation for an accurate sum of large arrays.
//! http://en.wikipedia.org/wiki/Kahan_summation_algorithm
////////////////////////////////////////////////////////////////////////////////
template <class T>
T reduceCPU(T *data, int size)
{
    T sum = data[0];
    T c = (T)0.0;

    for (int i = 1; i < size; i++) {
        T y = data[i] - c;
        T t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }

    return sum;
}

unsigned int nextPow2(unsigned int x)
{
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}

// Compute the number of threads and blocks to use for the given reduction kernel
void getNumBlocksAndThreads(int whichKernel, int n, int maxBlocks, int maxThreads, int &blocks, int &threads)
{
    // get device capability, to avoid block/grid size exceed the upper bound
    cudaDeviceProp prop;
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    if (whichKernel < 4) {
        threads = (n < maxThreads) ? nextPow2(n) : maxThreads;
        blocks = (n + threads - 1) / threads;
    }
    else {
        threads = (n < maxThreads * 2) ? nextPow2((n+1)/2) : maxThreads;
        blocks = (n + (threads * 2 - 1)) / (threads * 2);
    }

    if (((float)threads * blocks) > ((float)prop.maxGridSize[0] * prop.maxThreadsPerBlock)) {
        printf("n is too large, please choose a smaller number!\n");
    }

    if (blocks > prop.maxGridSize[0]) {
        printf(
            "Grid size <%d> exceeds the device capability <%d>, set block size as "
            "%d (original %d)\n",
            blocks, prop.maxGridSize[0], threads * 2, threads);

        blocks /= 2;
        threads *= 2;
    }
}