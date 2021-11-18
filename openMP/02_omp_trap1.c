/*****************************************************************************
 * File:        02_omp_trap1.c
 * Purpose:     Estimate definite integral (or area under curve) using 
 *              trapezoidal rule.
 * Compile:     gcc -Wall -fopenmp -o 02_omp_trap1 02_omp_trap1.c
 * Run:         ./02_omp_trap1 <number of threads>
 * 
 * Input:       a, b, n
 * Output:      estimate of integral from a to b of f(x) using n trapezoidals.
 *****************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void Usage(char* prog_name);
double f(double x); /* function we're integrating */
void Trap(double a, double b, int n, double* global_result_p);

int main(int argc, char* argv[])
{
    if (argc != 2)
        Usage(argv[0]);

    int thread_count = strtol(argv[1], NULL, 10);
    
    double a, b;
    int n;
    printf("Enter a, b, and n\n");
    scanf("%lf %lf %d", &a, &b, &n);
    if (n % thread_count != 0)
        Usage(argv[0]);

    double global_result = 0.0;
#pragma omp parallel num_threads(thread_count)
    Trap(a, b, n, &global_result);

    printf("With n = %d trapezoids, our estimate\n", n);
    printf("of the integral from %f to %f = %f\n", a, b, global_result);

    return 0;
}

/*****************************************************************************
 * Function:        Usage
 * Purpose:         Print a message indicating how program should be started
 *                  and terminate.
 *****************************************************************************/
void Usage(char* prog_name)
{
    fprintf(stderr, "Usage: %s <thread_count>\n", prog_name);
    fprintf(stderr, "   number of trapezoids must be evenly divisible by number of threads\n");
    exit(0);
}

/*****************************************************************************
 * Function:        f
 * Purpose:         Compute value of function to be integrated
 * Input arg:       x
 * Return val:      f(x)
 *****************************************************************************/
double f(double x)
{
    return x*x;
}

/*****************************************************************************
 * Function:        Trap
 * Purpose:         Use trapezoidal rule to estimate definite integral
 * Input arg:       
 *      a: left endpoint
 *      b: right endpoint
 *      n: number of trapezoids
 * Output arg:
 *      integral: estimate of integral from a to b of f(x)
 *****************************************************************************/
void Trap(double a, double b, int n, double* p_global_result)
{
    double h, local_a, local_b;
    int local_n;
    int my_rank = omp_get_thread_num();
    int thread_count = omp_get_num_threads();

    h = (b-a)/n;
    local_n = n/thread_count;
    local_a = a + my_rank*local_n*h;
    local_b = local_a + local_n*h;

    double my_result = (f(local_a) + f(local_b))/2.0;
    for (int i = 1; i < local_n; i++)
        my_result += f(local_a + i*h);
    my_result = my_result*h;

#pragma omp critical
    *p_global_result += my_result;
}