/*****************************************************************************
 * File:        10_omp_sin_sum.c
 * Purpose:     Compute a sum in which each term is the value of a function
 *              applied to a non-negative integer i and evaluation of the
 *              function requires work propotional to i.
 * Compile:     gcc -Wall -fopenmp -o 10_omp_sin_sum 10_omp_sin_sum.c -lm [-DDEBUG]
 * Run:         ./10_omp_sin_sum <number of threads> <number of terms>
 * 
 * Input:       none
 * Output:      sum of n terms and elapsed time to compute the sum
 * 
 * Note:
 *  1.  The computed sum is
 * 
 *          sin(0) + sin(1) + . . . + sin(n(n+3)/2)
 * 
 *  2.  The function f(i) is
 * 
 *          sin(i(i+1)/2) + sin(i(i+1)/2 + 1) + . . . + sin(i(i+1)/2 + i)
 * 
 *  3.  The parallel for directive uses a runtime schedule clause. So
 *      the environment variable OMP_SCHEDULE should be either
 *      "static, n/thread_count" for a block schedule or "static, 1" for
 *      a cyclic schedule
 *****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#ifdef DEBUG
int* iterations;
#endif

void Usage(char* prog_name);
double Sum(long n, int thread_count);
double Check_sum(long n, int thread_count);
double f(long i);
void Print_iters(int interations[], long n);

int main(int argc, char* argv[])
{
    if (argc != 3)
        Usage(argv[0]);
    int thread_count = strtol(argv[1], NULL, 10);
    long n = strtol(argv[2], NULL, 10);
#ifdef DEBUG
    iterations = (int*)malloc((n+1)*sizeof(int));
#endif

    double start, finish, global_result;
    start = omp_get_wtime();
    global_result = Sum(n, thread_count);
    finish = omp_get_wtime();

    double error, check;
    check = Check_sum(n, thread_count);
    error = fabs(global_result - check);

    printf("Result = %.14f\n", global_result);
    printf("Check = %.14f\n", check);
    printf("With n = %ld terms, the error is %.14f\n", n, error);
    printf("Elapsed time = %f seconds\n", finish - start);

#ifdef DEBUG
    Print_iters(iterations, n);
    free(iterations);
#endif
}

/*****************************************************************************
 * Function:        Usage
 * Purpose:         Print a message indicating how program should be started
 *                  and terminate.
 *****************************************************************************/
void Usage(char* prog_name)
{
    fprintf(stderr, "Usage: %s <number of threads> <number of terms>\n", prog_name);
    exit(0);
}

/*****************************************************************************
 * Function:        f
 * Purpose:         Compute value of function in which work is propotional to
 *                  the size of the first arg.
 * In args:         i
 * Return:          
 *      f(i) = sin(i(i+1)/2) + sin(i(i+1)/2 + 1) + ... + sin(i(i+1)/2 + i)
 *****************************************************************************/
double f(long i)
{
    long start = i*(i+1) / 2;
    long finish = start + i;
    double return_val = 0.0;

    for (long j = start; j <= finish; j++) {
        return_val += sin(j);
    }

    return return_val;
}

/*****************************************************************************
 * Function:        Sum
 * Purpose:         Find the sum of the terms f(0), f(1), ..., f(n) where
 *                  evaluating f requires work proportional to its argument
 * In args:         
 *      n:              number of terms
 *      thread_count:   number of threads
 * Return:          
 *      approx:         f(0) + f(1) + ... f(n)
 *****************************************************************************/
double Sum(long n, int thread_count)
{
    double approx = 0.0;

#pragma omp parallel for num_threads(thread_count) \
    reduction(+: approx) schedule(auto)
    for (int i = 0; i <= n; i++) {
        approx += f(i);
#ifdef DEBUG
        iterations[i] = omp_get_thread_num();
#endif
    }

    return approx;
}

/*****************************************************************************
 * Function:        Check_sum
 * Purpose:         Find the sum of the terms f(0), f(1), ..., f(n) by using
 *                  sin function in math library
 * In args:         
 *      n:              number of terms
 *      thread_count:   number of threads
 * Return:          
 *      check:          f(0) + f(1) + ... f(n)
 *****************************************************************************/
double Check_sum(long n, int thread_count)
{
    long i;
    long finish = n*(n+3) / 2;
    double check = 0.0;

#pragma omp parallel for num_threads(thread_count) \
    default(none) shared(n, finish) private(i) \
    reduction(+: check)
    for (i = 0; i <= finish; i++) {
        check += sin(i);
    }

    return check;
}

/*****************************************************************************
 * Function:        Print_iters
 * Purpose:         Print which thread was assigned which iteration.
 * In args:         
 *      iterations: iterations[i] = thread assigned iteration i
 *      n:          size of iterations array
 *****************************************************************************/
void Print_iters(int iterations[], long n)
{
    printf("\n");
    printf("Thread\t\tIterations\n");
    printf("------\t\t----------\n");
    int which_thread = iterations[0];
    int start_iter = 0, stop_iter = 0;
    for (int i = 0; i <= n; i++) {
        if (iterations[i] == which_thread) {
            stop_iter = i;
        }
        else {
            printf("%4d  \t\t%d -- %d\n", which_thread, start_iter, stop_iter);
            which_thread = iterations[i];
            start_iter = stop_iter = i;
        }
    }
    printf("%4d  \t\t%d -- %d\n", which_thread, start_iter, stop_iter);
}