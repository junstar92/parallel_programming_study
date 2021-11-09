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
    int dest;

    if (my_rank == 0) {
        printf("Enter a, b, and n\n");
        scanf("%lf %lf %d", p_a, p_b, p_n);
        for (dest = 1; dest < comm_sz; dest++) {
            MPI_Send(p_a, 1, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);
            MPI_Send(p_b, 1, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);
            MPI_Send(p_n, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
        } 
    } 
    else { /* my_rank != 0 */
        MPI_Recv(p_a, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD,
            MPI_STATUS_IGNORE);
        MPI_Recv(p_b, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD,
            MPI_STATUS_IGNORE);
        MPI_Recv(p_n, 1, MPI_INT, 0, 0, MPI_COMM_WORLD,
            MPI_STATUS_IGNORE);
    } 
}