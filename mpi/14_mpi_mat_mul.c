/*****************************************************************************
 * File:        14_mpi_mat_mul.c
 * Purpose:     Implement parallel matrix-matrix multiplication.
 *              One matrix is distributed by block rows and another matrix 
 *              is distributes by blocl columns by using traspose of it.
 * Compile:     mpicc -Wall -o 14_mpi_mat_mul 14_mpi_mat_mul.c
 * Run:         mpiexec -n <p> 14_mpi_mat_mul
 *                  - p: the number of processes
 * 
 * Input:       rows a1 and cols a2 of matrix A
 *              rows b1 and cols b2 of matrix B
 * Output:      C = AB (a1 x b2)
 *****************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

const int RMAX = 1000000;

void Check_for_error(int local_ok, char fname[], char message[], MPI_Comm comm);
void Get_dims(char dim1[], char dim2[], int* p_m, int* p_n, int my_rank, int comm_sz, MPI_Comm comm);
void Allocate_matrix(double** local_mat, int local_m, int n, MPI_Comm comm);
void Get_matrix(double local_mat[], int m, int local_m, int n, char mat_name[], int my_rank, MPI_Comm comm);
void Print_matrix(double local_mat[], int m, int local_m, int n, char title[], int my_rank, MPI_Comm comm);
void Print_matrix_trans(double local_mat[], int m, int local_m, int n, char title[], int my_rank, MPI_Comm comm);
void Mat_mul(double local_A[], double local_B[], double local_C[], int local_m, int n, int k, int local_k, MPI_Comm comm);

int main(void)
{
    double* local_A;
    double* local_BT;
    double* local_C;

    int a1, local_a1, a2;
    int bt1, local_bt1, bt2;
    int my_rank, comm_sz;
    MPI_Comm comm;

    srand(10);

    MPI_Init(NULL, NULL);
    comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &comm_sz);
    MPI_Comm_rank(comm, &my_rank);

    Get_dims("a1", "a2", &a1, &a2, my_rank, comm_sz, comm);
    Get_dims("b1", "b2", &bt2, &bt1, my_rank, comm_sz, comm);
    Check_for_error((int)(a2 == bt2), "main", 
        "dim2 of Matrix A and dim1 of Matrix B are not same", comm);
    local_a1 = a1 / comm_sz;
    local_bt1 = bt1 / comm_sz;

    Allocate_matrix(&local_A, local_a1, a2, comm);
    Allocate_matrix(&local_BT, bt1, bt2, comm);
    Allocate_matrix(&local_C, local_a1, bt1, comm);

    Get_matrix(local_A, a1, local_a1, a2, "A", my_rank, comm);
    //Print_matrix(local_A, a1, local_a1, a2, "A", my_rank, comm);
    Get_matrix(local_BT, bt1, local_bt1, bt2, "B_trans", my_rank, comm);
    //Print_matrix_trans(local_BT, bt1, local_bt1, bt2, "B", my_rank, comm);

    double start, finish, local_elapsed, elapsed;
    MPI_Barrier(comm);
    start = MPI_Wtime();
    Mat_mul(local_A, local_BT, local_C, local_a1, a2, bt1, local_bt1, comm);
    finish = MPI_Wtime();

    local_elapsed = finish - start;
    MPI_Reduce(&local_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

    //Print_matrix(local_C, a1, local_a1, bt1, "C", my_rank, comm);
    if (my_rank == 0)
        printf("Elapsed Time = %f seconds\n", elapsed);    

    free(local_A);
    free(local_BT);
    free(local_C);

    MPI_Finalize();

    return 0;
}

/*****************************************************************************
 * Function:        Check_for_error
 * Purpose:         Check whether any process has found an error.
 *                  If so, print message and terminate all processes.
 *                  Otherwise, continue execution.
 * Arguments:
 *  - local_ok:     1 if calling process has found an error, 0 otherwise
 *  - fname:        name of function calling Check_for_error
 *  - message:      message to print if there's error
 *  - comm:         communicator containing processes calling Check_for_error
 *                  It should be MPI_COMM_WORLD
 *****************************************************************************/
void Check_for_error(
    int         local_ok    /* in */,
    char        fname[]     /* in */,
    char        message[]   /* in */,
    MPI_Comm    comm        /* in */)
{
    int ok;

    MPI_Allreduce(&local_ok, &ok, 1, MPI_INT, MPI_MIN, comm);
    if (ok == 0)
    {
        int my_rank;
        MPI_Comm_rank(comm, &my_rank);
        if (my_rank == 0)
        {
            fprintf(stderr, "Proc %d > In %s, %s\n", my_rank, fname,
                    message);
            fflush(stderr);
        }
        MPI_Finalize();
        exit(-1);
    }
}

/*****************************************************************************
 * Function:        Get_dims
 * Purpose:         Get the dimensions of the matrix and the vectors from stdin
 * Arguments:
 *  - dim1:         names of rows
 *  - dim2:         names of cols
 *  - p_m:          global number of rows of matrix
 *  - p_n:          global number of cols of matrix
 *  - my_rank:      calling process' rank in comm
 *  - comm_sz:      number of processes in comm
 *  - comm:         communicator containing processes calling Get_dims
 *****************************************************************************/
void Get_dims(
    char        dim1[]     /* in  */,
    char        dim2[]     /* in  */,
    int*        p_m        /* out */,
    int*        p_n        /* out */,
    int         my_rank    /* in  */,
    int         comm_sz    /* in  */,
    MPI_Comm    comm       /* in  */)
{
    int local_ok = 1;

    if (my_rank == 0) {
        printf("Enter the number of rows %s\n", dim1);
        scanf("%d", p_m);
        printf("Enter the number of columns %s\n", dim2);
        scanf("%d", p_n);
    }

    MPI_Bcast(p_m, 1, MPI_INT, 0, comm);
    MPI_Bcast(p_n, 1, MPI_INT, 0, comm);

    if (*p_m <= 0 || *p_n <= 0 || (*p_m % comm_sz != 0) || (*p_n % comm_sz != 0))
        local_ok = 0;
    char message[100];
    sprintf(message, "%s and %s must be positive and evenly divisible by comm_sz", dim1, dim2);
    Check_for_error(local_ok, "Get_dims", message, comm);
}

/*****************************************************************************
 * Function:        Allocate_arrays
 * Purpose:         Allocate storage for local parts of A, x, and y
 * Arguments:
 *  - pp_local_mat: local storage for matrix (m/comm_sz x n)
 *  - local_m:      local number of rows of matrix
 *  - n:            global number of cols of matrix
 *  - comm:         communicator containing processes calling Allocate_arrays
 *****************************************************************************/
void Allocate_matrix(
    double**    pp_local_mat   /* in/out */,
    int         local_m        /* in     */,
    int         n              /* in     */,
    MPI_Comm    comm           /* in     */)
{
    int local_ok = 1;

    *pp_local_mat = (double*)malloc(local_m * n * sizeof(double));

    if (*pp_local_mat == NULL)
        local_ok = 0;
    Check_for_error(local_ok, "Allocate_matrix",
        "Can't allocate local matrix", comm);
}

/*****************************************************************************
 * Function:        Get_matrix
 * Purpose:         Get matrix from rand function on process 0 and distribute 
 *                  among the processes using a block row distribution
 * Arguments:
 *  - local_mat:    the local matrix
 *  - m:            global number of rows of matrix
 *  - local_m:      local number of rows of matrix A
 *  - n:            global and local number of cols of matrix A
 *  - mat_name:     description of matrix (e.g., "A")
 *  - my_rank:      calling process' rank in comm
 *  - comm:         communicator containing processes calling Get_matrix
 *****************************************************************************/
void Get_matrix(
    double      local_mat[] /* in/out */,
    int         m           /* in     */,
    int         local_m     /* in     */,
    int         n           /* in     */,
    char        mat_name[]  /* in     */,
    int         my_rank     /* in     */,
    MPI_Comm    comm        /* in     */)
{
    double* mat = NULL;
    int local_ok = 1;

    if (my_rank == 0) {
        mat = (double*)malloc(m * n * sizeof(double));
        if (mat == NULL)
            local_ok = 0;
        Check_for_error(local_ok, "Get_matrix",
            "Can't allocate temporary matrix", comm);
        
        printf("Get matrix %s from rand function...\n", mat_name);
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                mat[i*n + j] = (rand() % RMAX) / (RMAX/10.0); //rand() % 10;

        MPI_Scatter(mat, local_m * n, MPI_DOUBLE,
                local_mat, local_m * n, MPI_DOUBLE, 0, comm);
    }
    else {
        Check_for_error(local_ok, "Get_matrix",
            "Can't allocate temporary matrix", comm);
        MPI_Scatter(mat, local_m * n, MPI_DOUBLE,
                local_mat, local_m * n, MPI_DOUBLE, 0, comm);
    }
}

/*****************************************************************************
 * Function:        Print_matrix
 * Purpose:         Print a matrix distributed by block rows to stdout
 * Arguments:
 *  - local_mat:    calling process' part of matrix
 *  - m:            global number of rows
 *  - local_m:      local number of rows (m/comm_sz)
 *  - n:            global (and local) number of cols
 *  - title:        name of matrix
 *  - my_rank:      calling process' rank in comm
 *  - comm:         communicator containing processes calling Print_matrix
 *****************************************************************************/
void Print_matrix(
    double      local_mat[] /* in */,
    int         m           /* in */,
    int         local_m     /* in */,
    int         n           /* in */,
    char        title[]     /* in */,
    int         my_rank     /* in */,
    MPI_Comm    comm        /* in */)
{
    double* mat = NULL;
    int local_ok = 1;

    if (my_rank == 0) {
        mat = (double*)malloc(m * n * sizeof(double));
        if (mat == NULL)
            local_ok = 0;
        Check_for_error(local_ok, "Print_matrix",
            "Can't allocate temporary matrix", comm);
        
        MPI_Gather(local_mat, local_m * n, MPI_DOUBLE,
                mat, local_m * n, MPI_DOUBLE, 0, comm);
        
        printf("\nThe matrix %s\n", title);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++)
                printf("%f ", mat[i*n + j]);
            printf("\n");
        }
        printf("\n");
        free(mat);
    }
    else {
        Check_for_error(local_ok, "Print_matrix",
            "Can't allocate temporary matrix", comm);
        MPI_Gather(local_mat, local_m * n, MPI_DOUBLE,
                mat, local_m * n, MPI_DOUBLE, 0, comm);
    }
}

/*****************************************************************************
 * Function:        Print_matrix_trans
 * Purpose:         Print a tranpose of input matrix distributed by block rows to stdout
 * Arguments:
 *  - local_mat:    calling process' part of matrix
 *  - m:            global number of rows
 *  - local_m:      local number of rows (m/comm_sz)
 *  - n:            global (and local) number of cols
 *  - title:        name of matrix
 *  - my_rank:      calling process' rank in comm
 *  - comm:         communicator containing processes calling Print_matrix_trans
 *****************************************************************************/
void Print_matrix_trans(
    double      local_mat[] /* in */,
    int         m           /* in */,
    int         local_m     /* in */,
    int         n           /* in */,
    char        title[]     /* in */,
    int         my_rank     /* in */,
    MPI_Comm    comm        /* in */)
{
    double* mat = NULL;
    int local_ok = 1;

    if (my_rank == 0) {
        mat = (double*)malloc(m * n * sizeof(double));
        if (mat == NULL)
            local_ok = 0;
        Check_for_error(local_ok, "Print_matrix",
            "Can't allocate temporary matrix", comm);
        
        MPI_Gather(local_mat, local_m * n, MPI_DOUBLE,
                mat, local_m * n, MPI_DOUBLE, 0, comm);
        
        printf("\nThe matrix %s\n", title);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++)
                printf("%f ", mat[j*n + i]);
            printf("\n");
        }
        printf("\n");
        
        free(mat);
    }
    else {
        Check_for_error(local_ok, "Print_matrix",
            "Can't allocate temporary matrix", comm);
        MPI_Gather(local_mat, local_m * n, MPI_DOUBLE,
                mat, local_m * n, MPI_DOUBLE, 0, comm);
    }
}

/*****************************************************************************
 * Function:        Mat_mul
 * Purpose:         Implement parallel matrix-matrix multiplication
 * Arguments:
 *  - local_A:      calling process' rows of matrix A (m x n)
 *  - local_BT:     calling process' rows of tranpose of matrix B (n x k)
 *  - local_C:      calling process' local matrix to save result of multiplation
 *                  C = AB (m x k)
 *  - local_m:      calling process' number of rows in matrix A
 *  - n:            global cols of matrix A
 *  - k:            global cols of matrix B (== rows of matrix BT)
 *  - local_k:      calling process' number of cols in matrix B
 *                  (== local rows of matrix BT)
 *  - comm:         communicator containing processes calling Mat_mul
 *****************************************************************************/
void Mat_mul(double local_A[], double local_BT[], double local_C[], int local_m, int n, int k, int local_k, MPI_Comm comm)
{
    double* BT;
    int local_ok = 1;

    BT = (double*)malloc(k * n * sizeof(double));
    if (BT == NULL)
        local_ok = 0;
    Check_for_error(local_ok, "Mat_mul",
        "Can't allocate temporary matrix", comm);
    
    MPI_Allgather(local_BT, local_k * n, MPI_DOUBLE,
            BT, local_k * n, MPI_DOUBLE, comm);

    for (int mm = 0; mm < local_m; mm++) {
        for (int kk = 0; kk < k; kk++) {
            local_C[mm * k + kk] = 0.0;
            for (int nn = 0; nn < n; nn++) {
                local_C[mm * k + kk] += local_A[mm * n + nn] * BT[kk * n + nn];
            }
        }
    }

    free(BT);
}