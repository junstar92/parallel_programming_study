/*****************************************************************************
 * File:        07_omp_pi.c
 * Purpose:     Estimate pi using OpenMP and the formula
 *                  pi = 4*[1 - 1/3 + 1/5 - 1/7 + 1/9 - . . . ]
 * Compile:     gcc -Wall -fopenmp -o 07_omp_pi 07_omp_pi.c [-lm]
 * Run:         ./07_omp_pi <number of threads> <n>
 *              <n> is the number of terms of the series to use
 * 
 * Input:       none
 * Output:      The estimate of pi and the value of pi computed by the arctan
 *              function in the math library
 *****************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

void Usage(char* prog_name);

int main(int argc, char* argv[])
{
    int thread_count;
    long long n;

    if (argc != 3)
        Usage(argv[0]);
    thread_count = strtol(argv[1], NULL, 10);
    n = strtoll(argv[2], NULL, 10);
    if (thread_count < 1 || n < 1)
        Usage(argv[0]);

    double factor, sum = 0.0;
#pragma omp parallel for num_threads(thread_count) \
    reduction(+: sum) private(factor)
    for (int i = 0; i < n; i++) {
        factor = (i % 2 == 0) ? 1.0 : -1.0;
        sum += factor/(2*i + 1);
#ifdef DEBUG
    printf("Thread %d > i = %d, my_sum = %f\n", omp_get_thread_num(), i, sum);
#endif
    }

    sum = 4.0*sum;
    printf("With n = %lld terms and %d threads,\n", n, thread_count);
    printf("    Our estimate of pi = %.14f\n", sum);
    printf("                    pi = %.14f\n", 4.0*atan(1.0));

    return 0;
}

/*****************************************************************************
 * Function:        Usage
 * Purpose:         Print a message indicating how program should be started
 *                  and terminate.
 *****************************************************************************/
void Usage(char* prog_name)
{
    fprintf(stderr, "Usage: %s <thread_count> <n>\n", prog_name);
    fprintf(stderr, "   thread_count is the number of threads >= 1\n");
    fprintf(stderr, "   n is the number of terms and should be >= 1\n");
    exit(0);
}