/*****************************************************************************
 * File:        07_mpi_vec_add.c
 * Purpose:     Implement parallel vector addition using a block
 *              distribution of the vectors. This version illustrates
 *              use of MPI_Scatter and MPI_Gather
 * Compile:     mpicc -Wall -o 07_mpi_vec_add 07_mpi_vec_add.c
 * Run:         mpiexec -n <number of proesses> ./07_mpi_vec_add
 * 
 * Input:       The order of the vectors, n, should be evenly divisible
 * Output:      The sum vector z = x + y
 *****************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

const int RMAX = 100;

void Check_for_error(int local_ok, char fname[], char message[], MPI_Comm comm);
void Read_n(int* p_n, int* p_local_n, int my_rank, int comm_sz, MPI_Comm comm);
void Allocate_vectors(double** pp_local_x, double** pp_local_y, double** pp_local_z, int local_n, MPI_Comm comm);
void Get_vector(double local_vec[], int local_n, int n, char vec_name[], int my_rank, MPI_Comm comm);
void Read_vector(double local_vec[], int local_n, int n, char vec_name[], int my_rank, MPI_Comm comm);
void Print_vector(double local_vec[], int local_n, int n, char title[], int my_rank, MPI_Comm comm);
void Parallel_vector_sum(double local_x[], double local_y[], double local_z[], int local_n);

int main(void)
{
    int n, local_n, comm_sz, my_rank;
    double *local_x, *local_y, *local_z;
    MPI_Comm comm;

    srand(0);

    MPI_Init(NULL, NULL);
    comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &comm_sz);
    MPI_Comm_rank(comm, &my_rank);

    Read_n(&n, &local_n, my_rank, comm_sz, comm);
    Allocate_vectors(&local_x, &local_y, &local_z, local_n, comm);
    
    Get_vector(local_x, local_n, n, "x", my_rank, comm);
    Print_vector(local_x, local_n, n, "x is", my_rank, comm);
    Get_vector(local_y, local_n, n, "y", my_rank, comm);
    Print_vector(local_y, local_n, n, "y is", my_rank, comm);

    Parallel_vector_sum(local_x, local_y, local_z, local_n);
    Print_vector(local_z, local_n, n, "The sum is", my_rank, comm);

    free(local_x);
    free(local_y);
    free(local_z);

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
 * Function:        Read_n
 * Purpose:         Get the order of the vectors from stdin on proc 0 and
 *                  broadcast to other processes.
 * Arguments:
 *  - p_n:          global value of n
 *  - p_local_n:    local value of n = n/comm_sz
 *  - my_rank:      process rank in communicator
 *  - comm_sz:      number of processes in communicator
 *  - comm:         communicator containing all the calling processes
 *****************************************************************************/
void Read_n(
    int*        p_n         /* out */,
    int*        p_local_n   /* out */,
    int         my_rank     /* in  */,
    int         comm_sz     /* in  */,
    MPI_Comm    comm        /* in  */)
{
    int local_ok = 1;
    char* fname = "Read_n";

    if (my_rank == 0) {
        printf("What's the order of the vectors\n");
        scanf("%d", p_n);
    }

    MPI_Bcast(p_n, 1, MPI_INT, 0, comm);
    
    if ((*p_n <= 0) || ((*p_n % comm_sz) != 0))
        local_ok = 0;
    Check_for_error(local_ok, fname,
        "n should be > 0 and evenly divisible by comm_sz", comm);
    
    *p_local_n = *p_n / comm_sz;
}

/*****************************************************************************
 * Function:        Allocate_vectors
 * Purpose:         Allocate storage for x, y, and z
 * Arguments:
 *  - pp_local_x:   pointers to memory
 *  - pp_lcoal_y:   pointers to memory
 *  - pp_lcoal_z:   pointers to memory
 *  - local_n:      the size of the local vectors
 *  - comm:         communicator containing all the calling processes
 *****************************************************************************/
void Allocate_vectors(
    double**    pp_local_x  /* out */,
    double**    pp_local_y  /* out */,
    double**    pp_local_z  /* out */,
    int         local_n     /* in  */,
    MPI_Comm    comm        /* in  */)
{
    int local_ok = 1;
    char* fname = "Allocate_vectors";

    *pp_local_x = (double*)malloc(local_n * sizeof(double));
    *pp_local_y = (double*)malloc(local_n * sizeof(double));
    *pp_local_z = (double*)malloc(local_n * sizeof(double));

    if (*pp_local_x == NULL || *pp_local_y == NULL || *pp_local_z == NULL)
        local_ok = 0;
    Check_for_error(local_ok, fname,
        "Can't allocate local vector(s)", comm);
}

/*****************************************************************************
 * Function:        Get_vector
 * Purpose:         Get a vector from rand function on process 0 and
 *                  distribute among the processes using a block distribution.
 * Arguments:
 *  - local_vec:    local vector read
 *  - local_n:      size of local vectors
 *  - n:            size of global vectors
 *  - vec_name:     name of vector being read (e.g., "x")
 *  - my_rank:      calling process' rank in comm
 *  - comm:         communicator containing all the calling processes
 *****************************************************************************/
void Get_vector(
    double      local_vec[] /* out */,
    int         local_n     /* in  */,
    int         n           /* in  */,
    char        vec_name[]  /* in  */,
    int         my_rank     /* in  */,
    MPI_Comm    comm        /* in  */)
{
    double* tmp_vec = NULL;
    int local_ok = 1;
    char* fname = "Get_vector";

    if (my_rank == 0) {
        tmp_vec = (double*)malloc(n * sizeof(double));
        if (tmp_vec == NULL)
            local_ok = 0;
        Check_for_error(local_ok, fname, "Can't allocate temporary vector", comm);

        for (int i = 0; i < n; i++) {
            tmp_vec[i] = rand() % RMAX;
        }

        MPI_Scatter(tmp_vec, local_n, MPI_DOUBLE, local_vec, local_n, MPI_DOUBLE, 0, comm);
        free(tmp_vec);
    }
    else {
        Check_for_error(local_ok, fname, "Can't allocate temporary vector", comm);
        MPI_Scatter(tmp_vec, local_n, MPI_DOUBLE, local_vec, local_n, MPI_DOUBLE, 0, comm);
    }
}

/*****************************************************************************
 * Function:        Print_vector
 * Purpose:         Print a vector that has a block distribution to stdout
 * Arguments:
 *  - local_vec:    local vector read
 *  - local_n:      size of local vectors
 *  - n:            size of global vectors
 *  - title:        title to precede print out
 *  - my_rank:      calling process' rank in comm
 *  - comm:         communicator containing all the calling processes
 *****************************************************************************/
void Print_vector(
    double      local_vec[] /* in */,
    int         local_n     /* in */,
    int         n           /* in */,
    char        title[]     /* in */,
    int         my_rank     /* in */,
    MPI_Comm    comm        /* in */)
{
    double* tmp_vec;
    int local_ok = 1;
    char* fname = "Print_vector";

    if (my_rank == 0) {
        tmp_vec = (double*)malloc(n * sizeof(double));
        if (tmp_vec == NULL)
            local_ok = 0;
        Check_for_error(local_ok, fname, "Can't allocate temporary vector", comm);

        MPI_Gather(local_vec, local_n, MPI_DOUBLE, tmp_vec, local_n, MPI_DOUBLE, 0, comm);

        printf("%s\n", title);
        for (int i = 0 ; i < n; i++) {
            printf("%f ", tmp_vec[i]);
        }
        printf("\n");
        free(tmp_vec);
    }
    else {
        Check_for_error(local_ok, fname, "Can't allocate temporary vector", comm);
        MPI_Gather(local_vec, local_n, MPI_DOUBLE, tmp_vec, local_n, MPI_DOUBLE, 0, comm);
    }
}

/*****************************************************************************
 * Function:        Parallel_vector_sum
 * Purpose:         Add a vector that's been distributed among the processes
 * Arguments:
 *  - local_x:      local storage of one of the vector being added
 *  - local_y:      local storage for the second vector being added
 *  - local_z:      local storage for the sum of the two vectors
 *  - local_n:      size of local vectors
 *****************************************************************************/
void Parallel_vector_sum(
    double  local_x[]   /* in  */,
    double  local_y[]   /* in  */,
    double  local_z[]   /* out */,
    int     local_n     /* in  */)
{
    for (int local_i = 0; local_i < local_n; local_i++)
        local_z[local_i] = local_x[local_i] + local_y[local_i];
}