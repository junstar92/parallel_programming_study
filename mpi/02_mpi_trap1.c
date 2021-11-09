/*
 * File:        02_mpi_trap1.c
 * Purpose:     Use MPI to implement a parallel version of the trapezoidal rule.
 *              In this version the endpoints of the interval and
 *              the number of trapezoids are hardwired.
 * Compile:     mpicc -Wall -o 02_mpi_trap1 02_mpi_trap1.c
 * Run:         mpiexec -n <number of proesses> ./02_mpi_trap1
 * 
 * Algorithm:
 *    1.  Each process calculates "its" interval of
 *        integration.
 *    2.  Each process estimates the integral of f(x)
 *        over its interval using the trapezoidal rule.
 *    3a. Each process != 0 sends its integral to 0.
 *    3b. Process 0 sums the calculations received from
 *        the individual processes and prints the result.
 */
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

double Trap(double a, double b, int n, double h);
double f(double x);

int main(void)
{
    int my_rank, comm_sz, n = 1024, local_n;
    double a = 0.0, b = 3.0, h, local_a, local_b, local_int, total_int;

    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    h = (b-a)/n;
    local_n = n/comm_sz;

    local_a = a + my_rank*local_n*h;
    local_b = local_a + local_n*h;
    local_int = Trap(local_a, local_b, local_n, h);

    if (my_rank != 0) {
        MPI_Send(&local_int, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }
    else {
        total_int = local_int;
        for (int source = 1; source < comm_sz; source++) {
            MPI_Recv(&local_int, 1, MPI_DOUBLE, source, 0,
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            total_int += local_int;
        }
    }

    if (my_rank == 0) {
        printf("With n = %d trapezoids, our estimate\n", n);
        printf("of the integral from %f to %f = %.15f\n", a, b, total_int);
    }

    MPI_Finalize();

    return 0;
}

double Trap(double a, double b, int n, double h)
{
    double integral;
    
    integral = (f(a) + f(b)) / 2.0;

    for(int k = 0; k < n; k++) {
        integral += f(a + k*h);
    }
    integral = integral * h;

    return integral;
}

double f(double x)
{
    return x*x;
}