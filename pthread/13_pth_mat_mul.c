/*****************************************************************************
 * File:        13_pth_mat_mul.c
 * Purpose:     Compute a parallel matrix-matrix product. Matrix is distributed 
 *              by block rows.
 * Compile:     gcc -Wall -o 13_pth_mat_mul 13_pth_mat_mul.c -pthread [-DDEBUG]
 * Run:         ./13_pth_mat_mul <thread_count> <m> <n> <k> <sol>
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
#include <pthread.h>
#include <sys/time.h>

#define GET_TIME(now) { \
    struct timeval t; \
    gettimeofday(&t, NULL); \
    now = t.tv_sec + t.tv_usec/1000000.0; \
}

const int RMAX = 1000000;
#ifdef DEBUG
const int NCOUNT = 1; // number of multiplication
#else
const int NCOUNT = 100; // number of multiplication
#endif
int thread_count, m, n, k, sol;
double *A, *B, *C, *BT;

void Get_args(int argc, char* argv[]);
void Usage(char* prog_name);
void Generate_matrix(double mat[], int m, int n);
void Transpose_matrix(double mat[], double mat_t[], int m, int n);
void Print_matrix(double mat[], int m, int n, char* title);

void* Pth_mat_mul1(void* rank);
void* Pth_mat_mul2(void* rank);

int main(int argc, char* argv[])
{
    Get_args(argc, argv);

    pthread_t* thread_handles = (pthread_t*)malloc(thread_count * sizeof(pthread_t));
    
    A = (double*)malloc(m * n * sizeof(double));
    B = (double*)malloc(n * k * sizeof(double));
    C = (double*)malloc(m * k * sizeof(double));
    BT = (double*)malloc(k * n * sizeof(double));

    Generate_matrix(A, m, n);
    Generate_matrix(B, n, k);
#ifdef DEBUG
    Print_matrix(A, m, n, "A");
    Print_matrix(B, m, n, "B");
#endif

    void* sol_function;
    switch (sol) {
    case 1:
        sol_function = &Pth_mat_mul1;
        break;
    case 2:
        sol_function = &Pth_mat_mul2;
        break;
    }
    
    double start, finish, avg_elapsed = 0.0;
    for (int count = 0; count < NCOUNT; count++) {
        GET_TIME(start);
        if (sol == 2) {
            Transpose_matrix(B, BT, n, k);
        }
        for (long thread = 0; thread < thread_count; thread++)
            pthread_create(&thread_handles[thread], NULL, sol_function, (void*)thread);
        for (long thread = 0; thread < thread_count; thread++)
            pthread_join(thread_handles[thread], NULL);
        GET_TIME(finish);

        printf("[%3d] Elapsed time = %.6f seconds\n", count+1, finish-start);
        avg_elapsed += (finish - start) / NCOUNT;
    }

#ifdef DEBUG
    Print_matrix(C, m, k, "The product is");
#endif
    
    printf("Average elapsed time = %.6f seconds\n", avg_elapsed);

    free(A);
    free(B);
    free(C);
    free(BT);
    free(thread_handles);

    return 0;
}


/*****************************************************************************
 * Function:        Get_args
 * Purpose:         Get and check command list arguments
 * In args:         argc, argv
 *****************************************************************************/
void Get_args(int argc, char* argv[])
{
    if (argc != 6)
        Usage(argv[0]);
    
    thread_count = strtol(argv[1], NULL, 10);
    m = strtol(argv[2], NULL, 10);
    n = strtol(argv[3], NULL, 10);
    k = strtol(argv[4], NULL, 10);
    sol = strtol(argv[5], NULL, 10);
    if (thread_count <= 0 || m <= 0 || n <= 0 || k <= 0)
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
 * In arg:          mat, m, n, rank
 * Out arg:         mat_t
 *****************************************************************************/
void Transpose_matrix(double mat[], double mat_t[], int m, int n)
{
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
 * Function:        Pth_mat_mul1
 * Purpose:         Multiply an m x n matrix by an n x k matrix
 * In args:         rank
 * Global In args:  A, B, m, n, k, thread_count
 * Global Out args: C
 *****************************************************************************/
void* Pth_mat_mul1(void* rank)
{
    long my_rank = (long)rank;
    int local_m = m / thread_count;
    int first_row = my_rank * local_m;
    int last_row = first_row + local_m;

#ifdef DEBUG
    printf("Thread %ld > local_m = %d\n", my_rank, local_m);
#endif
    double temp;
    for (int i = first_row; i < last_row; i++) {
        for (int j = 0; j < k; j++) {
            temp = 0.0;
            for (int l = 0; l < n; l++) {
                temp += A[i*n + l] * B[l*k + j];
            }
            C[i*k + j] = temp;
        }
    }

    return NULL;
}

/*****************************************************************************
 * Function:        Pth_mat_mul2
 * Purpose:         Multiply an m x n matrix by an n x k matrix's transposition
 *                  to avoid cache miss
 * In args:         rank
 * Global In args:  A, BT, m, n, k, thread_count
 * Global Out args: C
 *****************************************************************************/
void* Pth_mat_mul2(void* rank)
{
    long my_rank = (long)rank;
    int local_m = m / thread_count;
    int first_row = my_rank * local_m;
    int last_row = first_row + local_m;

#ifdef DEBUG
    printf("Thread %ld > local_m = %d\n", my_rank, local_m);
#endif
    double temp;
    for (int i = first_row; i < last_row; i++) {
        for (int j = 0; j < k; j++) {
            temp = 0.0;
            for (int l = 0; l < n; l++) {
                temp += A[i*n + l] * BT[j*n + l];
            }
            C[i*k + j] = temp;
        }
    }

    return NULL;
}