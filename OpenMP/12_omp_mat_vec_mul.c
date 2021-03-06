/*****************************************************************************
 * File:        12_omp_mat_vec_mul.c
 * Purpose:     Computes a parallel matrix-vector product. Matrix is distributed
 *              by block rows. Vectors are distributed by blocks.
 *              Matrix A and Vector x are generated by random number generator.
 * Compile:     gcc -Wall -fopenmp -o 12_omp_mat_vec_mul 12_omp_mat_vec_mul.c
 *              [-DDEBUG]
 * Run:         ./12_omp_mat_vec_mul <number of threads> <m> <n>
 * 
 * Input:       A, x
 * Output:      
 *              y: the product vector, y = Ax
 *              Elapsed time for the computation
 * 
 * Note:        1.  Storage for A, x, y is dynamically allocated.
 *              2.  Number of threads(thread_count) should evenly divide both 
 *                  m and n. The program doesn't check for this.
 *              3.  Distribution of A, x, and y is logical: all three are 
 *                  globally shared.
 *****************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

const int RMAX = 1000000;

void Get_args(int argc, char* argv[], int* thread_count, int* m, int* n);
void Usage(char* prog_name);
void Generate_matrix(double mat[], int m, int n);
void Generate_vector(double vec[], int n);
void Print_matrix(double mat[], int m, int n, char* title);
void Print_vector(double vec[], int n, char* title);

void Omp_mat_vec_mul(double A[], double x[], double y[], int m, int n, int thread_count);

int main(int argc, char* argv[])
{
    int thread_count, m, n;
    Get_args(argc, argv, &thread_count, &m, &n);

    double *A, *x, *y;
    A = (double*)malloc(m * n * sizeof(double));
    x = (double*)malloc(n * sizeof(double));
    y = (double*)malloc(m * sizeof(double));

    Generate_matrix(A, m, n);
#ifdef DEBUG
    Print_matrix(A, m, n, "A");
#endif
    Generate_vector(x, n);
#ifdef DEBUG
    Print_vector(x, n, "x");
#endif

    Omp_mat_vec_mul(A, x, y, m, n, thread_count);

#ifdef DEBUG
    Print_vector(y, m, "The product is");
#endif

    free(A);
    free(x);
    free(y);

    return 0;
}

/*****************************************************************************
 * Function:        Get_args
 * Purpose:         Get and check command list arguments
 * In args:         argc, argv
 * Out args:        thread_count, m, n
 *****************************************************************************/
void Get_args(int argc, char* argv[], int* thread_count, int* m, int* n)
{
    if (argc != 4)
        Usage(argv[0]);
    
    *thread_count = strtol(argv[1], NULL, 10);
    *m = strtol(argv[2], NULL, 10);
    *n = strtol(argv[3], NULL, 10);
    if (*thread_count <= 0 || *m <= 0 || *n <= 0)
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
    fprintf(stderr, "Usage: %s <thread_count> <m> <n>\n", prog_name);
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
    srand(m);
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            mat[i*n + j] = (rand() % RMAX) / (RMAX / 10.0);
}

/*****************************************************************************
 * Function:        Generate_vector
 * Purpose:         Generate vector entries by using the random number generator
 * In arg:          n
 * Out arg:         vec
 *****************************************************************************/
void Generate_vector(double vec[], int n)
{
    srand(n);
    for (int i = 0; i < n; i++)
        vec[i] = (rand() % RMAX) / (RMAX / 10.0);
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
 * Function:        Print_vector
 * Purpose:         Print the vector
 * In args:         vec, n, title
 *****************************************************************************/
void Print_vector(double vec[], int n, char* title)
{
    printf("%s\n", title);
    for (int i = 0; i < n; i++)
        printf("%f ", vec[i]);
    printf("\n");
}

/*****************************************************************************
 * Function:        Omp_mat_vec_mul
 * Purpose:         Multiply an m x n matrix by an n x 1 column vector
 * In args:         A, x, m, n, thread_count
 * Out arg:         y
 *****************************************************************************/
void Omp_mat_vec_mul(double A[], double x[], double y[], int m, int n, int thread_count)
{
    double start, finish, temp;
    start = omp_get_wtime();
#pragma omp parallel for num_threads(thread_count) \
    default(none) private(temp) shared(A, x, y, m, n)
    for (int i = 0; i < m; i++) {
        y[i] = 0.0;
        for (int j = 0; j < n; j++) {
            //y[i] += A[i*n + j] * x[j];
            temp = A[i*n + j] * x[j];
            y[i] += temp;
        }
    }
    finish = omp_get_wtime();

    printf("Elapsed time = %f seconds\n", finish - start);
}