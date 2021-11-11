/*****************************************************************************
 * File:        08_mpi_mat_vec_mul.c
 * Purpose:     Implement parallel matrix-vector multiplication using
 *              one-dimensional arrays to store the vectors and the matrix.
 *              Vectors use block distributions and the matrix is distributed
 *              by block rows.
 * Compile:     mpicc -Wall -o 08_mpi_mat_vec_mul 08_mpi_mat_vec_mul.c
 * Run:         mpiexec -n <number of proesses> ./08_mpi_mat_vec_mul
 * 
 * Input:       Dimension of the matrix (m = number of rows,
 *                                       n = number of columns)
 *              m x n matrix A
 *              n-dimensional vector x
 * Output:      Vector y = Ax
 *****************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

const int RMAX = 10000000;

void Check_for_error(int local_ok, char fname[], char message[], MPI_Comm comm);
void Get_dims(int* p_m, int* p_local_m, int* p_n, int* p_local_n, int my_rank, int comm_sz, MPI_Comm comm);
void Allocate_arrays(double** pp_local_A, double** pp_local_x, double** pp_local_y, int local_m, int n, int local_n, MPI_Comm comm);
void Get_matrix(double local_mat[], int m, int local_m, int n, char mat_name[], int my_rank, MPI_Comm comm);
void Get_vector(double local_vec[], int n, int local_n, char vec_name[], int my_rank, MPI_Comm comm);
void Print_matrix(double local_mat[], int m, int local_m, int n, char title[], int my_rank, MPI_Comm comm);
void Print_vector(double local_vec[], int n, int local_n, char title[], int my_rank, MPI_Comm comm);
void Mat_vec_mul(double local_A[], double local_x[], double local_y[], int local_m, int n, int local_n, MPI_Comm comm);


int main(void)
{
    double *local_A, *local_x, *local_y;
    int m, local_m, n, local_n;
    int my_rank, comm_sz;
    MPI_Comm comm;

    MPI_Init(NULL, NULL);
    comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &comm_sz);
    MPI_Comm_rank(comm, &my_rank);

    Get_dims(&m, &local_m, &n, &local_n, my_rank, comm_sz, comm);
    Allocate_arrays(&local_A, &local_x, &local_y, local_m, n, local_n, comm);

    Get_matrix(local_A, m, local_m, n, "A", my_rank, comm);
    Print_matrix(local_A, m, local_m, n, "A", my_rank, comm);
    Get_vector(local_x, n, local_n, "x", my_rank, comm);
    Print_vector(local_x, n, local_n, "x", my_rank, comm);

    Mat_vec_mul(local_A, local_x, local_y, local_m, n, local_n, comm);

    Print_vector(local_y, m, local_m, "y", my_rank, comm);

    free(local_A);
    free(local_x);
    free(local_y);

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
    if (ok == 0) {
        int my_rank;
        MPI_Comm_rank(comm, &my_rank);
        if (my_rank == 0) {
            fprintf(stderr, "Proc %d > In %s, %s\n", my_rank, fname, message);
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
 *  - p_m:          global number of rows of A and components of y
 *  - p_local_m:    local number of rows of A and compoenents of y
 *  - p_n:          global number of cols of A and components of x
 *  - p_local_n     local number of cols of A and components of x
 *  - my_rank:      calling process' rank in comm
 *  - comm_sz:      number of processes in comm
 *  - comm:         communicator containing processes calling Get_dims
 *****************************************************************************/
void Get_dims(
    int*        p_m         /* out */,
    int*        p_local_m   /* out */,
    int*        p_n         /* out */,
    int*        p_local_n   /* out */,
    int         my_rank     /* in  */,
    int         comm_sz     /* in  */,
    MPI_Comm    comm        /* in  */)
{
    int local_ok = 1;

    if (my_rank == 0) {
        printf("Enter the number of rows\n");
        scanf("%d", p_m);
        printf("Enter the number of columns\n");
        scanf("%d", p_n);
    }

    MPI_Bcast(p_m, 1, MPI_INT, 0, comm);
    MPI_Bcast(p_n, 1, MPI_INT, 0, comm);
    
    if ((*p_m <= 0) || (*p_n <= 0) || (*p_m % comm_sz != 0) ||
            (*p_n % comm_sz != 0))
        local_ok = 0;
    Check_for_error(local_ok, "Get_dims", 
        "m and n must be positive and evenly divisible by comm_sz", comm);

    *p_local_m = *p_m / comm_sz;
    *p_local_n = *p_n / comm_sz;
}

/*****************************************************************************
 * Function:        Allocate_arrays
 * Purpose:         Allocate storage for local parts of A, x, and y
 * Arguments:
 *  - pp_local_A:   local storage for matrix (m/comm_sz x n)
 *  - pp_lcoal_x:   local storage for x (n/comm_sz components)
 *  - pp_lcoal_y:   local storage for y (m/comm_sz components)
 *  - local_m:      local number of rows of A and components of y
 *  - n:            global and local number of cols of A and global number of
 *                  components x
 *  - local_n:      local number of components of x
 *  - comm:         communicator containing processes calling Allocate_arrays
 *****************************************************************************/
void Allocate_arrays(
    double**    pp_local_A  /* out */,
    double**    pp_local_x  /* out */,
    double**    pp_local_y  /* out */,
    int         local_m     /* in  */,
    int         n           /* in  */,
    int         local_n     /* in  */,
    MPI_Comm    comm        /* in  */)
{
    int local_ok = 1;

    *pp_local_A = (double*)malloc(local_m * n * sizeof(double));
    *pp_local_x = (double*)malloc(local_n * sizeof(double));
    *pp_local_y = (double*)malloc(local_m * sizeof(double));

    if ((*pp_local_A == NULL) || (*pp_local_x == NULL) || (*pp_local_y == NULL))
        local_ok = 0;
    Check_for_error(local_ok, "Allocate_arrays",
        "Can't allocate local arrays", comm);
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
 *  - comm:         communicator containing processes calling Allocate_arrays
 *****************************************************************************/
void Get_matrix(
    double      local_mat[] /* out */,
    int         m           /* in  */,
    int         local_m     /* in  */,
    int         n           /* in  */,
    char        mat_name[]  /* in  */,
    int         my_rank     /* in  */,
    MPI_Comm    comm        /* in  */)
{
    double* mat = NULL;
    int local_ok = 1;

    if (my_rank == 0) {
        mat = (double*)malloc(m * n * sizeof(double));
        
        if (mat == NULL)
            local_ok = 0;
        Check_for_error(local_ok, "Get_matrix",
            "Can't allocate temporary matrix", comm);

        printf("Get the matrix %s from rand function\n", mat_name);
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                mat[i*n + j] = (rand() % RMAX) / (RMAX/10.0);

        MPI_Scatter(mat, local_m * n, MPI_DOUBLE,
                local_mat, local_m * n, MPI_DOUBLE, 0, comm);
        free(mat);
    }
    else {
        Check_for_error(local_ok, "Get_matrix",
            "Can't allocate temporary matrix", comm);
        MPI_Scatter(mat, local_m * n, MPI_DOUBLE,
                local_mat, local_m * n, MPI_DOUBLE, 0, comm);
    }
}

/*****************************************************************************
 * Function:        Get_vector
 * Purpose:         Get a vector from rand function on process 0 and
 *                  distribute among the processes using a block distribution.
 * Arguments:
 *  - local_vec:    local vector read
 *  - n:            size of global vectors
 *  - local_n:      size of local vectors
 *  - vec_name:     name of vector being read (e.g., "x")
 *  - my_rank:      calling process' rank in comm
 *  - comm:         communicator containing processes calling Get_vector
 *****************************************************************************/
void Get_vector(
    double      local_vec[] /* out */,
    int         n           /* in  */,
    int         local_n     /* in  */,
    char        vec_name[]  /* in  */,
    int         my_rank     /* in  */,
    MPI_Comm    comm        /* in  */)
{
    double* vec = NULL;
    int local_ok = 1;

    if (my_rank == 0) {
        vec = (double*)malloc(n * sizeof(double));
        if (vec == NULL)
            local_ok = 0;
        Check_for_error(local_ok, "Get_vector",
            "Can't allocate temporary vector", comm);

        printf("Get the vector %s from rand function\n", vec_name);
        for (int i = 0; i < n; i++) {
            vec[i] = (rand() % RMAX) / (RMAX/10.0);
        }

        MPI_Scatter(vec, local_n, MPI_DOUBLE,
                local_vec, local_n, MPI_DOUBLE, 0, comm);
        free(vec);
    }
    else {
        Check_for_error(local_ok, "Get_vector",
            "Can't allocate temporary vector", comm);
        MPI_Scatter(vec, local_n, MPI_DOUBLE, 
                local_vec, local_n, MPI_DOUBLE, 0, comm);
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
 * Function:        Print_vector
 * Purpose:         Print a vector that has a block distribution to stdout
 * Arguments:
 *  - local_vec:    calling process' part of vector
 *  - n:            global number of components
 *  - local_n:      local number of components (n/comm_sz)
 *  - title:        name of vector
 *  - my_rank:      calling process' rank in comm
 *  - comm:         communicator containing processes calling Print_vector
 *****************************************************************************/
void Print_vector(
    double      local_vec[] /* in */,
    int         n           /* in */,
    int         local_n     /* in */,
    char        title[]     /* in */,
    int         my_rank     /* in */,
    MPI_Comm    comm        /* in */)
{
    double* vec;
    int local_ok = 1;

    if (my_rank == 0) {
        vec = (double*)malloc(n * sizeof(double));
        if (vec == NULL)
            local_ok = 0;
        Check_for_error(local_ok, "Print_vector",
            "Can't allocate temporary vector", comm);

        MPI_Gather(local_vec, local_n, MPI_DOUBLE,
                vec, local_n, MPI_DOUBLE, 0, comm);

        printf("\nThe vector %s\n", title);
        for (int i = 0 ; i < n; i++) {
            printf("%f ", vec[i]);
        }
        printf("\n");
        free(vec);
    }
    else {
        Check_for_error(local_ok, "Print_vector",
            "Can't allocate temporary vector", comm);
        MPI_Gather(local_vec, local_n, MPI_DOUBLE,
                vec, local_n, MPI_DOUBLE, 0, comm);
    }
}

/*****************************************************************************
 * Function:        Mat_vec_mul
 * Purpose:         Print a vector that has a block distribution to stdout
 * Arguments:
 *  - local_A:      calling process' rows of matrix A
 *  - local_x:      calling process' components of vector x
 *  - local_y:      calling process' local vector to save result of multiplation
 *                  y = Ax
 *  - local_m:      calling process' number of rows
 *  - n:            global (and local) number of columns
 *  - local_n:      calling process' number of components of x
 *  - comm:         communicator containing processes calling Mat_vec_mul
 *****************************************************************************/
void Mat_vec_mul(
    double      local_A[]   /* in  */,
    double      local_x[]   /* in  */,
    double      local_y[]   /* out */,
    int         local_m     /* in  */,
    int         n           /* in  */,
    int         local_n     /* in  */,
    MPI_Comm    comm        /* in  */)
{
    double* x;
    int local_ok = 1;

    x = (double*)malloc(n * sizeof(double));
    if (x == NULL)
        local_ok = 0;
    Check_for_error(local_ok, "Mat_vec_mul",
        "Can't allocate temporary vector", comm);
    
    MPI_Allgather(local_x, local_n, MPI_DOUBLE,
        x, local_n, MPI_DOUBLE, comm);
    
    for (int local_i = 0; local_i < local_m; local_i++) {
        local_y[local_i] = 0.0;
        for (int j = 0; j < n; j++)
            local_y[local_i] += local_A[local_i * n + j] * x[j];
    }
    free(x);
}