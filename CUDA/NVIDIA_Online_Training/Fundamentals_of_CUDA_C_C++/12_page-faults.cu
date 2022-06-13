// 12_page-faults.cu

__global__
void deviceKernel(int *a, const int N)
{
    int idx = blockIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < N; i += stride) {
        a[i] = i;
    }
}

void hostFunction(int *a, const int N)
{
    for (int i = 0; i < N; i++) {
        a[i] = i;
    }
}

int main()
{
    int N = 2 << 24;
    size_t size = N * sizeof(int);

    int *a;
    cudaMallocManaged(&a, size);

    /*
     * Conduct experiments to learn more about the behavior of
     * `cudaMallocManaged`.
     *
     * What happens when unified memory is accessed only by the GPU?
     *   deviceKernel(a, N);
     *   cudaDeviceSynchronize();
     * What happens when unified memory is accessed only by the CPU?
     *   hostFunction<<<256, 256>>>(a, N);
     *   cudaDeviceSynchronize();
     * What happens when unified memory is accessed first by the GPU then the CPU?
     *   deviceKernel<<<256, 256>>>(a, N)
     *   cudaDeviceSynchronize();
     *   hostFunction(a, N);
     * What happens when unified memory is accessed first by the CPU then the GPU?
     *   hostFunction(a, N);
     *   deviceKernel<<<256, 256>>>(a, N);
     *   cudaDeviceSynchronize();
     *
     * Hypothesize about UM behavior, page faulting specificially, before each
     * experiment, and then verify by running `nsys`.
     */
    

    cudaFree(a);
}