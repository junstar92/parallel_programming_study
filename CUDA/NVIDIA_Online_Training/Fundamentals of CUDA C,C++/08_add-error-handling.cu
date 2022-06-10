// 08_add-error-handling.cu
#include <stdio.h>

void init(int *a, const int N)
{
    for (int i = 0; i < N; i++) {
        a[i] = i;
    }
}

__global__
void doubleElements(int *a, const int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = idx; i < N + stride; i += stride) {
        a[i] *= 2;
    }
}

bool checkElementsAreDoubled(int *a, const int N)
{
    for (int i = 0; i < N; i++) {
        if (a[i] != i * 2)
            return false;
    }

    return true;
}

int main()
{
    int N = 1000;
    int *a;

    size_t size = N * sizeof(int);

    // Use 'cudaMallocManaged' to allocate pointer 'a' available
    // on both the host and the device.
    cudaMallocManaged(&a, size);

    init(a, N);

    size_t threads_per_block = 1024;
    size_t number_of_blocks = 32;

    cudaError_t syncErr, asyncErr;

    doubleElements<<<number_of_blocks, threads_per_block>>>(a, N);

    // catch errors for both the kernel launch above and any errors that
    // occur during the asynchronous 'doubleElements' kernel execution.
    syncErr = cudaGetLastError();
    asyncErr = cudaDeviceSynchronize();

    // print errors should they exist.
    if (syncErr != cudaSuccess)
        printf("Error(sync): %s\n", cudaGetErrorString(syncErr));
    if (asyncErr != cudaSuccess)
        printf("Error(async): %s\n", cudaGetErrorString(asyncErr));

    bool areDoubled = checkElementsAreDoubled(a, N);
    printf("All elements were doubled? %s\n", areDoubled ? "TRUE" : "FALSE");

    // Use 'cudaFree' to free memory allocated with 'cudaMallocManaged'
    cudaFree(a);
}