/*
 * File:        03_mpi_output.c
 * Purpose:     A program in which multiple MPI processes try to print a message.
 * Compile:     mpicc -Wall -o 03_mpi_output 03_mpi_output.c
 * Run:         mpiexec -n <number of proesses> ./03_mpi_output
 */
#include <stdio.h>
#include <mpi.h>

int main(void)
{
    int my_rank, comm_sz;

    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    printf("Proc %d of %d > Done anyone have a toothpick?\n", my_rank, comm_sz);

    MPI_Finalize();

    return 0;
}