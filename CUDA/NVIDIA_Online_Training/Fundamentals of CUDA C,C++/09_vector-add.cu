// 09_vector-add.cu
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

void initWith(float num, float* a, const int N)
{
    for (int i = 0; i < N; i++) {
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

    float *a, *b, *c;

    checkCuda(cudaMallocManaged(&a, size));
    checkCuda(cudaMallocManaged(&b, size));
    checkCuda(cudaMallocManaged(&c, size));

    initWith(3, a, N);
    initWith(4, b, N);
    initWith(0, c, N);

    size_t threadsPerBlock = 256;
    size_t numberOfBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    addVectorsInto<<<numberOfBlocks, threadsPerBlock>>>(c, a, b, N);

    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());

    checkElementsAre(7, c, N);

    checkCuda(cudaFree(a));
    checkCuda(cudaFree(b));
    checkCuda(cudaFree(c));
}