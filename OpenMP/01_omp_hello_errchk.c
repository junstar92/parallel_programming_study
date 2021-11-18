/*****************************************************************************
 * File:        01_omp_hello_errchk.c
 * Purpose:     A parallel hello, world program that uses OpenMP
 * Compile:     gcc -Wall -fopenmp -o 01_omp_hello_errchk 01_omp_hello_errchk.c
 * Run:         ./01_omp_hello_errchk <number of threads>
 * 
 * Input:       none
 * Output:      A message from each thread
 *****************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#ifdef _OPENMP
#include <omp.h>
#endif

void Usage(char* prog_name);
void Hello(int thread_count);   /* thread function */

int main(int argc, char* argv[])
{
#ifdef _OPENMP
    printf("_OPENMP = %d\n", _OPENMP);
#endif
    int thread_count;

    if (argc != 2)
        Usage(argv[0]);
    thread_count = strtol(argv[1], NULL, 10);
    if (thread_count <= 0)
        Usage(argv[0]);

#pragma omp parallel num_threads(thread_count)
    Hello(thread_count);

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
    fprintf(stderr, "   thread_count should be positive\n");
    exit(0);
}

/*****************************************************************************
 * Function:        Hello
 * Purpose:         Thread function that prints message
 *****************************************************************************/
void Hello(int thread_count)
{
#ifdef _OPENMP
    int my_rank = omp_get_thread_num();
    int actual_thread_count = omp_get_num_threads();
#else
    int my_rank = 0;
    int actual_thread_count = 1;
#endif

    if (my_rank == 0 && thread_count != actual_thread_count)
        fprintf(stderr, "Number of threads started != %d\n", thread_count);
    printf("Hello from thread %d of %d\n", my_rank, actual_thread_count);
}