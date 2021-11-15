/*****************************************************************************
 * File:        01_pth_mat_vec_mul.c
 * Purpose:     Compute a parallel matrix-vector product. Matrix is distributed 
 *              by block rows. Vectors are distributed by block.
 * Compile:     gcc -Wall -o 01_pth_mat_vec_mul 01_pth_mat_vec_mul.c -pthread
 * Run:         ./01_pth_mat_vec_mul <thread_count>
 * 
 * Input:       m, n: dimension of matrix
 *              A, x: the matrix and the vector to be multiplied
 * Output:      y: the product vector
 * 
 * Node:        1.  Number of threads (thread_count) double evenly divided both
 *                  m and n. The program doesn't check for this.
 *              2.  This program uses a 1-dimensional array for A and compute 
 *                  subscripts using the formula A[i][j] = A[i*n + j]
 *              3.  Distribution of A, x, and y is logical: all three are
 *                  globally shared.
 *****************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>

/* Global variables */
const int RMAX = 1000000;
int thread_count;
int m, n;
double *A, *x, *y;

void Generate_matrix(double mat[], int m, int n);
void Print_matrix(double mat[], int m, int n, char* title);
void Generate_vector(double vec[], int n);
void Print_vector(double vec[], int n, char* title);

void* Pth_mat_vec_mul(void* rank);

int main(int argc, char* argv[])
{
    pthread_t* thread_handles;

    if (argc != 4) {
        fprintf(stderr, "usage: %s <thread_count> <m> <n>\n", argv[0]);
        exit(0);
    }

    thread_count = strtol(argv[1], NULL, 10);
    m = strtol(argv[2], NULL, 10);
    n = strtol(argv[3], NULL, 10);

    thread_handles = (pthread_t*)malloc(thread_count * sizeof(pthread_t));
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

    clock_t start = clock();

    for (long thread = 0; thread < thread_count; thread++)
        pthread_create(&thread_handles[thread], NULL, Pth_mat_vec_mul, (void*)thread);
    
    for (long thread = 0; thread < thread_count; thread++)
        pthread_join(thread_handles[thread], NULL);
    
    clock_t finish = clock();

#ifdef DEBUG
    Print_vector(y, m, "y");
#endif
    printf("Elasped time = %.6f seconds\n", (finish - start)/(double)CLOCKS_PER_SEC);

    free(A);
    free(x);
    free(y);
    free(thread_handles);

    return 0;
}

/*****************************************************************************
 * Function:        Generate_matrix
 * Purpose:         Generate elements of matrix from rand function
 * In args:         m, n
 * Out args:        A 
 *****************************************************************************/
void Generate_matrix(double A[], int m, int n)
{
    srand(m);
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            A[i*n + j] = (rand() % RMAX) / (double)(RMAX/10);
}

/*****************************************************************************
 * Function:        Print_matrix
 * Purpose:         Print matrix to stdout
 * In args:         A, m, n, title
 *****************************************************************************/
void Print_matrix(double A[], int m, int n, char* title)
{
    printf("\nThe matrix %s\n", title);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++)
            printf("%f ", A[i*n + j]);
        printf("\n");
    }
}

/*****************************************************************************
 * Function:        Generate_vector
 * Purpose:         Generate elements of vector from rand function
 * In args:         n
 * OUt args:        x
 *****************************************************************************/
void Generate_vector(double x[], int n)
{
    srand(n);
    for (int i = 0; i < n; i++)
        x[i] = (rand() % RMAX) / (double)(RMAX/10);
}

/*****************************************************************************
 * Function:        Print_vector
 * Purpose:         Print vector to stdout
 * In args:         x, n, title
 *****************************************************************************/
void Print_vector(double x[], int n, char* title)
{
    printf("\nThe vector %s\n", title);
    for (int i = 0 ; i < n; i++)
        printf("%f ", x[i]);
    printf("\n");
}

/*****************************************************************************
 * Function:        Pth_mat_vec_mul
 * Purpose:         Multiply an m x n matrix by an n x 1 columns vector
 * In args:         x, n, title
 *****************************************************************************/
void* Pth_mat_vec_mul(void* rank)
{
    long my_rank = (long)rank;
    int local_m = m / thread_count;
    int my_first_row = my_rank*local_m;
    int my_last_row = my_first_row + local_m;

#ifdef DEBUG
    printf("Thread %ld > local_m = %d\n", my_rank, local_m);
#endif

    for (int i = my_first_row; i < my_last_row; i++) {
        y[i] = 0.0;
        for (int j = 0; j < n; j++) {
            y[i] += A[i*n + j] * x[j];
        }
    }

    return NULL;
}