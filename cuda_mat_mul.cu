/*****************************************************************************
 * File:        cuda_mat_mul.cu
 * Purpose:     Compute a matrix-matrix product by using CUDA library.
 * Compile:     nvcc -o cuda_mat_mul cuda_mat_mul.cu -lcuda
 * Run:         ./cuda_mat_mul <m> <n> <k>
 *                  <m> : the rows of matrix A
 *                  <n> : the columns of matrix A and the rows of matrix B
 *                  <k> : the columns of matrix B
 * 
 * Input:       A, B
 * Output:      
 *              C: the product matrix, C = AB
 *              Elapsed time each multiplication and average elapsed time of
 *              100 multiplications
 *****************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda_runtime.h>

#define GET_TIME(now) { \
    struct timeval t; \
    gettimeofday(&t, NULL); \
    now = t.tv_sec + t.tv_usec/1000000.0; \
}

#define CUDA_CHECK(val) { \
	if (val != cudaSuccess) { \
		fprintf(stderr, "Error %s at line %d in file %s\n", cudaGetErrorString(val), __LINE__, __FILE__); \
		exit(1); \
	} \
}

const int RMAX = 1000000;
#ifdef DEBUG
const int NCOUNT = 1; // number of multiplication
#else
const int NCOUNT = 100; // number of multiplication
#endif

void Get_args(int argc, char* argv[], int* m, int* n, int* k);
void Usage(char* prog_name);
void Generate_matrix(double mat[], int m, int n);
void Print_matrix(double mat[], int m, int n, char* title);
__global__ void cuda_mat_mul(double *A, double *B, double *C, int m, int n, int k);

int main(int argc, char* argv[])
{
    int m, n, k;
    Get_args(argc, argv, &m, &n, &k);

    double *A, *B, *C;
    A = (double*)malloc(m * n * sizeof(double));
    B = (double*)malloc(n * k * sizeof(double));
    C = (double*)malloc(m * k * sizeof(double));

    Generate_matrix(A, m, n);
    Generate_matrix(B, n, k);
#ifdef DEBUG
    Print_matrix(A, m, n, "A");
    Print_matrix(B, n, k, "B");
#endif

    // Allocate the device input matrixs for A, B, C;
    double* d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void**)&d_A, m * n * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_B, n * k * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_C, m * k * sizeof(double)));

    dim3 dimGrid(ceil(n / 16.0), ceil(m / 16.0));
    dim3 dimBlock(16, 16); // 256 threads

    double start, finish, avg_elapsed = 0.0;
    // Launch the Matrix Multiplication CUDA Kernel
    for (int count = 0; count < NCOUNT; count++) {
        GET_TIME(start);
        // Copy the host matrixs A and B in host memory to the device matrixs in device memory
        CUDA_CHECK(cudaMemcpy(d_A, A, m * n * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, B, n * k * sizeof(double), cudaMemcpyHostToDevice));

        cuda_mat_mul<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, m, n, k);
        CUDA_CHECK(cudaGetLastError());
        // Copy the device result matrix in device memory to the host result matrix in host memory
        CUDA_CHECK(cudaMemcpy(C, d_C, m * k * sizeof(double), cudaMemcpyDeviceToHost));
        GET_TIME(finish);

        printf("[%3d] Elapsed time = %.6f seconds\n", count+1, finish-start);
        avg_elapsed += (finish - start) / NCOUNT;
    }

#ifdef DEBUG
    Print_matrix(C, m, k, "The product is");
#endif

    printf("Average elapsed time = %.6f seconds\n", avg_elapsed);

    // Free device global memory
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    free(A);
    free(B);
    free(C);

    return 0;
}

/*****************************************************************************
 * Function:        Get_args
 * Purpose:         Get and check command list arguments
 * In args:         argc, argv
 * Out args:        m, n, k
 *****************************************************************************/
void Get_args(int argc, char* argv[], int* m, int* n, int* k)
{
    if (argc != 4)
        Usage(argv[0]);
    
    *m = strtol(argv[1], NULL, 10);
    *n = strtol(argv[2], NULL, 10);
    *k = strtol(argv[3], NULL, 10);
    if (*m <= 0 || *n <= 0 || *k <= 0)
        Usage(argv[0]);
}

/*****************************************************************************
 * Function:        Usage
 * Purpose:         Print a message indicating how program should be started
 *                  and terminate.
 * In arg:          prog_name
 *****************************************************************************/
void Usage(char* prog_name)
{
    fprintf(stderr, "Usage: %s <m> <n> <k>\n", prog_name);
    exit(0);
}

/*****************************************************************************
 * Function:        Generate_matrix
 * Purpose:         Generate matrix entries by using the random number generator
 * In args:         m, n
 * Out arg:         mat
 *****************************************************************************/
void Generate_matrix(double mat[], int m, int n)
{
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            mat[i*n + j] = (rand() % RMAX) / (RMAX / 10.0);
}

/*****************************************************************************
 * Function:        Print_matrix
 * Purpose:         Print the matrix
 * In args:         mat, m, n, title
 *****************************************************************************/
void Print_matrix(double mat[], int m, int n, char* title)
{
    printf("%s\n", title);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++)
            printf("%f ", mat[i*n + j]);
        printf("\n");
    }
}

__global__ void cuda_mat_mul(double *A, double *B, double *C, int m, int n, int k)
{
    int ROW = blockIdx.x * blockDim.x + threadIdx.x;
    int COL = blockIdx.y * blockDim.y + threadIdx.y;

    if (ROW < m && COL < k) {
        double value = 0.0;
        for (int i = 0; i < k; i++) {
            value += A[ROW * n + i] * B[i * k + COL];
        }
        C[ROW * k + COL] = value;
    }

    __syncthreads();
}