/*****************************************************************************
 * File:        14_omp_mat_mul.c
 * Purpose:     Computes a parallel matrix-matrix product. Matrix is distributed
 *              by block rows.
 * Compile:     gcc -Wall -fopenmp -o 14_omp_mat_mul 14_omp_mat_mul.c
 *              [-DDEBUG]
 * Run:         ./14_omp_mat_mul <number of threads> <m> <n> <k> <sol>
 *                  <m> : the rows of matrix A
 *                  <n> : the columns of matrix A and the rows of matrix B
 *                  <k> : the columns of matrix B
 *                <sol> : number of solution
 *                  - 1 : Matrix A multiply by matrix B
 *                  - 2 : Matrix A multiply by transpose of matrix B
 * 
 * Input:       A, B
 * Output:      
 *              C: the product matrix, C = AB
 *              Elapsed time each multiplication and average elapsed time of
 *              100 multiplications
 *****************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

const int RMAX = 1000000;
const int NCOUNT = 100; // number of multiplication

void Get_args(int argc, char* argv[], int* thread_count, int* m, int* n, int* k, int* sol);
void Usage(char* prog_name);
void Generate_matrix(double mat[], int m, int n);
void Transpose_matrix(double mat[], double mat_t[], int m, int n, int thread_count);
void Print_matrix(double mat[], int m, int n, char* title);

void Omp_mat_mul1(double A[], double B[], double C[], int m, int n, int k, int thread_count);
void Omp_mat_mul2(double A[], double B[], double C[], double BT[], int m, int n, int k, int thread_count);

int main(int argc, char* argv[])
{
    int thread_count, m, n, k, sol;
    Get_args(argc, argv, &thread_count, &m, &n, &k, &sol);

    double *A, *B, *C, *BT;
    A = (double*)malloc(m * n * sizeof(double));
    B = (double*)malloc(n * k * sizeof(double));
    C = (double*)malloc(m * k * sizeof(double));
    BT = (double*)malloc(k * n * sizeof(double));

    Generate_matrix(A, m, n);
    Generate_matrix(B, n, k);
#ifdef DEBUG
    Print_matrix(A, m, n, "A");
    Print_matrix(B, n, k, "B");
#endif

    switch (sol) {
    case 1:
        Omp_mat_mul1(A, B, C, m, n, k, thread_count);
        break;
    case 2:
        Omp_mat_mul2(A, B, C, BT, m, n, k, thread_count);
        break;
    }

#ifdef DEBUG
    Print_matrix(C, m, k, "The product is");
#endif

    free(A);
    free(B);
    free(C);
    free(BT);

    return 0;
}

/*****************************************************************************
 * Function:        Get_args
 * Purpose:         Get and check command list arguments
 * In args:         argc, argv
 * Out args:        thread_count, m, n, k, sol
 *****************************************************************************/
void Get_args(int argc, char* argv[], int* thread_count, int* m, int* n, int* k, int* sol)
{
    if (argc != 6)
        Usage(argv[0]);
    
    *thread_count = strtol(argv[1], NULL, 10);
    *m = strtol(argv[2], NULL, 10);
    *n = strtol(argv[3], NULL, 10);
    *k = strtol(argv[4], NULL, 10);
    *sol = strtol(argv[5], NULL, 10);
    if (*thread_count <= 0 || *m <= 0 || *n <= 0 || *k <= 0)
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
    fprintf(stderr, "Usage: %s <thread_count> <m> <n> <k> <sol>\n", prog_name);
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
 * Function:        Transpose_matrix
 * Purpose:         Transpose matrix
 * In arg:          mat, m, n
 * Out arg:         mat_t
 *****************************************************************************/
void Transpose_matrix(double mat[], double mat_t[], int m, int n, int thread_count)
{
#pragma omp parallel for num_threads(thread_count) \
    default(none) shared(mat, mat_t, m, n)
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            mat_t[j*m + i] = mat[i*n + j];
        }
    }
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

/*****************************************************************************
 * Function:        Omp_mat_vec_mul1
 * Purpose:         Multiply an m x n matrix by an n x k matrix
 * In args:         A, B, m, n, k, thread_count
 * Out arg:         C
 *****************************************************************************/
void Omp_mat_mul1(double A[], double B[], double C[], int m, int n, int k, int thread_count)
{
    double start, finish, temp, avg_elapsed = 0.0;
    for (int count = 0; count < NCOUNT; count++) {
        start = omp_get_wtime();
#pragma omp parallel for num_threads(thread_count) \
    default(none) private(temp) shared(A, B, C, m, n, k)
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < k; j++) {
                temp = 0.0;
                for (int l = 0; l < n; l++) {
                    temp += A[i*n + l] * B[l*k + j];
                }
                C[i*k + j] = temp;
            }
        }
        finish = omp_get_wtime();
        printf("[%3d] Elapsed time = %f seconds\n", count+1, finish - start);
        avg_elapsed += (finish-start) / NCOUNT;
    }

    printf("Average elapsed time : %.6f seconds\n", avg_elapsed);
}

/*****************************************************************************
 * Function:        Omp_mat_mul2
 * Purpose:         Multiply an m x n matrix by an n x k matrix's transposition
 *                  to avoid cache miss
 * In args:         A, B, m, n, k, thread_count
 * Out arg:         C, BT
 *****************************************************************************/
void Omp_mat_mul2(double A[], double B[], double C[], double BT[], int m, int n, int k, int thread_count)
{
    double start, finish, temp, avg_elapsed = 0.0;
    for (int count = 0; count < NCOUNT; count++) {
        start = omp_get_wtime();
        Transpose_matrix(B, BT, n, k, thread_count);
#pragma omp parallel for num_threads(thread_count) \
    default(none) private(temp) shared(A, BT, C, m, n, k)
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < k; j++) {
                temp = 0.0;
                for (int l = 0; l < n; l++) {
                    temp += A[i*n + l] * BT[j*n + l];
                }
                C[i*k + j] = temp;
            }
        }
        finish = omp_get_wtime();
        printf("[%3d] Elapsed time = %f seconds\n", count+1, finish - start);
        avg_elapsed += (finish-start) / NCOUNT;
    }

    printf("Average elapsed time : %.6f seconds\n", avg_elapsed);
}