/*****************************************************************************
 * File:        06_omp_fibo.c
 * Purpose:     Try to compute n Fibonacci number using OpenMP.
 *              Show what happens if we try to parallelize a loop with
 *              dependences among the iterations. The program has a serious bug.
 * Compile:     gcc -Wall -fopenmp -o 06_omp_fibo 06_omp_fibo.c
 * Run:         ./06_omp_fibo <number of threads> <number of Fibonacci numbers>
 * 
 * Input:       none
 * Output:      A list of Fibonacci numbers
 * 
 * Note:        If your output seems to be OK, try increasing the number of 
 *              threads and/or n.
 *****************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void Usage(char* prog_name);

int main(int argc, char* argv[])
{
    int thread_count, n;
    long long* fibo;

    if (argc != 3)
        Usage(argv[0]);
    
    thread_count = strtol(argv[1], NULL, 10);
    n = strtol(argv[2], NULL, 10);

    fibo = (long long*)malloc(n * sizeof(long long));
    fibo[0] = fibo[1] = 1;

#pragma omp parallel for num_threads(thread_count)
    for (int i = 2; i < n; i++)
        fibo[i] = fibo[i-1] + fibo[i-2];
    
    printf("The first n Fibonacci numbers:\n");
    for (int i = 0; i < n; i++)
        printf("%d\t%lld\n", i, fibo[i]);
    
    free(fibo);

    return 0;
}

/*****************************************************************************
 * Function:        Usage
 * Purpose:         Print a message indicating how program should be started
 *                  and terminate.
 *****************************************************************************/
void Usage(char* prog_name)
{
    fprintf(stderr, "Usage: %s <thread_count> <number of Fibonacci numbers\n", prog_name);
    exit(0);
}