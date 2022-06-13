// 15_vector-add-manual-alloc.cu
#include <stdio.h>
#include <assert.h>

inline cudaError_t checkCuda(cudaError_t result)
{
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
    return result;
}

__global__
void initWith(float num, float *a, int N)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < N; i += stride)
    {
        a[i] = num;
    }
}

__global__
void addVectorsInto(float* result, float* a, float* b, const int N)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < N; i += stride) {
        result[i] = a[i] + b[i];
    }
}

void checkElementsAre(float target, float* array, const int N)
{
    for (int i = 0; i < N; i++) {
        if (array[i] != target) {
            printf("FAIL: array[%d] - %0.0f does not equal %0.0f\n", i, array[i], target);
            exit(1);
        }
    }
    printf("SUCCESS! All values added correctly.\n");
}

int main()
{
    const int N = 2 << 20;
    size_t size = N * sizeof(float);

    int deviceId;
    checkCuda(cudaGetDevice(&deviceId));

    cudaDeviceProp props;
    checkCuda(cudaGetDeviceProperties(&props, deviceId));

    float *a, *b, *c, *h_c;

    checkCuda(cudaMalloc(&a, size));
    checkCuda(cudaMalloc(&b, size));
    checkCuda(cudaMalloc(&c, size));
    checkCuda(cudaMallocHost(&h_c, size));

    size_t threadsPerBlock = props.maxThreadsPerBlock;
    size_t numberOfBlocks = props.multiProcessorCount;

    cudaStream_t stream1, stream2, stream3;
    checkCuda(cudaStreamCreate(&stream1));
    checkCuda(cudaStreamCreate(&stream2));
    checkCuda(cudaStreamCreate(&stream3));

    initWith<<<numberOfBlocks, threadsPerBlock, 0, stream1>>>(3, a, N);
    initWith<<<numberOfBlocks, threadsPerBlock, 0, stream2>>>(4, b, N);
    initWith<<<numberOfBlocks, threadsPerBlock, 0, stream3>>>(0, c, N);

    addVectorsInto<<<numberOfBlocks, threadsPerBlock>>>(c, a, b, N);

    checkCuda(cudaMemcpy(h_c, c, size, cudaMemcpyDeviceToHost));
    
    checkElementsAre(7, h_c, N);

    checkCuda(cudaStreamDestroy(stream1));
    checkCuda(cudaStreamDestroy(stream2));
    checkCuda(cudaStreamDestroy(stream3));

    checkCuda(cudaFree(a));
    checkCuda(cudaFree(b));
    checkCuda(cudaFree(c));
    checkCuda(cudaFreeHost(h_c));
}