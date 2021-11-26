/*****************************************************************************
 * File:        cublas_mat_mul.cu
 * Purpose:     Compute a matrix-matrix product by using CUDA BLAS library.
 * Compile:     nvcc -o cublas_mat_mul cublas_mat_mul.cu -lcublas
 * Run:         ./cublas_mat_mul <m> <n> <k>
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
#include <cublas.h>

#define GET_TIME(now) { \
    struct timeval t; \
    gettimeofday(&t, NULL); \
    now = t.tv_sec + t.tv_usec/1000000.0; \
}

#define CUBLAS_CHECK(val) { \
	if (val != CUBLAS_STATUS_SUCCESS) { \
		fprintf(stderr, "Error at line %d in file %s\n", __LINE__, __FILE__); \
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
    CUBLAS_CHECK(cublasAlloc(m*n, sizeof(double), (void**)&d_A));
    CUBLAS_CHECK(cublasAlloc(n*k, sizeof(double), (void**)&d_B));
    CUBLAS_CHECK(cublasAlloc(m*k, sizeof(double), (void**)&d_C));

    double start, finish, avg_elapsed = 0.0;
    // Launch the Matrix Multiplication CUDA Kernel
    CUBLAS_CHECK(cublasInit());
    for (int count = 0; count < NCOUNT; count++) {
        GET_TIME(start);
        // Copy the host matrixs A and B in host memory to the device matrixs in device memory
        CUBLAS_CHECK(cublasSetVector(m * n, sizeof(double), A, 1, d_A, 1));
        CUBLAS_CHECK(cublasSetVector(n * k, sizeof(double), B, 1, d_B, 1));

        cublasDgemm('T', 'T', m, k, n, 1.0, d_A, n, d_B, k, 0, d_C, m);
        CUBLAS_CHECK(cublasGetError());
        // Copy the device result matrix in device memory to the host result matrix in host memory
        CUBLAS_CHECK(cublasGetVector(m * k, sizeof(double), d_C, 1, C, 1));
        GET_TIME(finish);

        printf("[%3d] Elapsed time = %.6f seconds\n", count+1, finish-start);
        avg_elapsed += (finish - start) / NCOUNT;
    }

#ifdef DEBUG
    Print_matrix(C, m, k, "The product is");
#endif

    printf("Average elapsed time = %.6f seconds\n", avg_elapsed);

    // Free device global memory
    CUBLAS_CHECK(cublasFree(d_A));
    CUBLAS_CHECK(cublasFree(d_B));
    CUBLAS_CHECK(cublasFree(d_C));

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