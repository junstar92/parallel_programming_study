/*
 * File:        05_mpi_trap3.c
 * Purpose:     Use MPI to implement a parallel version of the trapezoidal rule.
 *              This version uses collective communications to distribute
 *              the input data and compute the global sum.
 * Compile:     mpicc -Wall -o 05_mpi_trap3 05_mpi_trap3.c
 * Run:         mpiexec -n <number of proesses> ./05_mpi_trap3
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
void Get_input(int my_rank, int comm_sz, double* p_a, double* p_b, int* p_n);

int main(void)
{
    int my_rank, comm_sz, n, local_n;
    double a, b, h, local_a, local_b, local_int, total_int;

    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    Get_input(my_rank, comm_sz, &a, &b, &n);

    h = (b-a)/n;
    local_n = n/comm_sz;

    local_a = a + my_rank*local_n*h;
    local_b = local_a + local_n*h;
    local_int = Trap(local_a, local_b, local_n, h);

    MPI_Reduce(&local_int, &total_int, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

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

void Get_input(int my_rank, int comm_sz, double* p_a, double* p_b, int* p_n)
{
    if (my_rank == 0) {
        printf("Enter a, b, and n\n");
        scanf("%lf %lf %d", p_a, p_b, p_n);
    }

    MPI_Bcast(p_a, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(p_b, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(p_n, 1, MPI_INT, 0, MPI_COMM_WORLD);
}