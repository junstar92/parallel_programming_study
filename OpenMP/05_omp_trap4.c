/*****************************************************************************
 * File:        05_omp_trap4.c
 * Purpose:     Estimate definite integral (or area under curve) using 
 *              trapezoidal rule. This version uses a parallel for directive.
 * Compile:     gcc -Wall -fopenmp -o 05_omp_trap4 05_omp_trap4.c
 * Run:         ./05_omp_trap4 <number of threads>
 * 
 * Input:       a, b, n
 * Output:      estimate of integral from a to b of f(x) using n trapezoidals.
 * 
 * Note:        In this version, it's not necessary for n to be evenly divisible
 *              by thread_count
 *****************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void Usage(char* prog_name);
double f(double x); /* function we're integrating */
double Trap(double a, double b, int n, int thread_count);

int main(int argc, char* argv[])
{
    if (argc != 2)
        Usage(argv[0]);

    int thread_count = strtol(argv[1], NULL, 10);
    
    double a, b;
    int n;
    printf("Enter a, b, and n\n");
    scanf("%lf %lf %d", &a, &b, &n);

    double global_result = 0.0;
    global_result = Trap(a, b, n, thread_count);

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
 * Purpose:         Use trapezoidal rule to estimate part of a definite
 *                  integral
 * Input arg:       
 *      a: left endpoint
 *      b: right endpoint
 *      n: number of trapezoids
 * Return: 
 *      approx: estimate of integral from a to b of f(x)
 *****************************************************************************/
double Trap(double a, double b, int n, int thread_count)
{
    double h, approx;

    h = (b-a)/n;
    approx = (f(a) + f(b))/2.0;
#pragma omp parallel for num_threads(thread_count) \
    reduction(+: approx)
    for (int i = 1; i < n; i++)
        approx += f(a + i*h);
    approx = approx*h;

    return approx;
}