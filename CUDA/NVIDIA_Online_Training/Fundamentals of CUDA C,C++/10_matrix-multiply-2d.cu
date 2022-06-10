// 10_matrix-multiply-2d.cu
#include <stdio.h>

#define N 64

__global__
void matrixMulGPU(int* a, int* b, int* c)
{
    int val = 0;

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < N && col < N) {
        for (int k = 0; k < N; k++) {
            val += a[row * N + k] * b[k * N + col];
        }
        c[row * N + col] = val;
    }
}

void matrixMulCPU(int* a, int* b, int* c)
{
    int val = 0;

    for (int row = 0; row < N; row++) {
        for (int col = 0; col < N; col++) {
            val = 0;
            for (int k = 0; k < N; k++) {
                val += a[row * N + k] * b[k * N + col];
            }
            c[row * N + col] = val;
        }
    }
}

int main()
{
    int *a, *b, *c_cpu, *c_gpu;

    size_t size = N * N * sizeof(int); // The number of bytes of an N x N matrix

    // Allocate Memory
    cudaMallocManaged(&a, size);
    cudaMallocManaged(&b, size);
    cudaMallocManaged(&c_cpu, size);
    cudaMallocManaged(&c_gpu, size);

    // Initialize Memory
    for (int row = 0; row < N; row++) {
        for (int col = 0; col < N; col++) {
            a[row * N + col] = row;
            b[row * N + col] = col + 2;
            c_cpu[row * N + col] = 0;
            c_gpu[row * N + col] = 0;
        }
    }

    // configuration
    dim3 threads_per_block(16, 16, 1); // A 16 x 16 block threads
    dim3 number_of_blocks((N / threads_per_block.x) + 1, (N / threads_per_block.y) + 1, 1);

    matrixMulGPU<<<number_of_blocks, threads_per_block>>>(a, b, c_gpu);

    cudaDeviceSynchronize();

    // Call the CPU version to check
    matrixMulCPU(a, b, c_cpu);

    // Compare the two answers
    bool error = false;
    for (int row = 0; row < N; row++) {
        for (int col = 0; col < N; col++) {
            if (c_cpu[row * N + col] != c_gpu[row * N + col]) {
                printf("FOUND ERROR at c[%d][%d]\n", row, col);
                error = true;
                break;
            }
        }
    }

    if (!error) {
        printf("Success!\n");
    }

    // Free all allocated memory
    cudaFree(a);
    cudaFree(b);
    cudaFree(c_cpu);
    cudaFree(c_gpu);
}