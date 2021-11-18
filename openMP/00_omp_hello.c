/*****************************************************************************
 * File:        00_omp_hello.c
 * Purpose:     A parallel hello, world program that uses OpenMP
 * Compile:     gcc -Wall -fopenmp -o 00_omp_hello 00_omp_hello.c
 * Run:         ./00_omp_hello <number of threads>
 * 
 * Input:       none
 * Output:      A message from each thread
 *****************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

/* thread function */
void Hello(void);

int main(int argc, char* argv[])
{
    int thread_count = strtol(argv[1], NULL, 10);

#pragma omp parallel num_threads(thread_count)
    Hello();

    return 0;
}

/*****************************************************************************
 * Function:        Hello
 * Purpose:         Thread function that prints message
 *****************************************************************************/
void Hello(void)
{
    int my_rank = omp_get_thread_num();
    int thread_count = omp_get_num_threads();

    printf("Hello from thread %d of %d\n", my_rank, thread_count);
}