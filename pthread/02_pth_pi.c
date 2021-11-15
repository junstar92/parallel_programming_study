/*****************************************************************************
 * File:        02_pth_pi.c
 * Purpose:     Estimate pi using serise
 * 
 *                  pi = 4*[1 - 1/3 + 1/5 - 1/7 + 1/9 - . . .]
 *              
 *              This version has a 'very serious bug'
 * 
 * Compile:     gcc -Wall -o 02_pth_pi 02_pth_pi.c -pthread [-lm]
 * Run:         ./02_pth_pi <number of threads> <n>
 *              <n>:the number of terms of the Maclarin series. It should be
 *                  evenly divisible by the number of threads
 * 
 * Input:       none
 * Output:      The estimate of pi using multiple threads, one thread, and the
 *              value computed by the math library arctan function.
 *              Also elapsed times for the multithreaded and singlethreaded 
 *              computations.
 *****************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <pthread.h>

#define GET_TIME(now) { \
    struct timeval t; \
    gettimeofday(&t, NULL); \
    now = t.tv_sec + t.tv_usec/1000000.0; \
}

const int MAX_THREADS = 1024;

/* global variables */
long thread_count;
long long n;
double sum;

void *Thread_sum(void* rank);

void Get_args(int argc, char* argv[]);
double Serial_pi(long long n);

int main(int argc, char* argv[])
{
    pthread_t* thread_handles;

    Get_args(argc, argv);

    thread_handles = (pthread_t*)malloc(thread_count * sizeof(pthread_t));
    sum = 0.0;

    for (long thread = 0; thread < thread_count; thread++)
        pthread_create(&thread_handles[thread], NULL, Thread_sum, (void*)thread);
    
    for (long thread = 0; thread < thread_count; thread++)
        pthread_join(thread_handles[thread], NULL);

    sum *= 4.0;

    printf("With n = %lld terms,\n", n);
    printf("   Multi-threaded estimate of pi  = %.15f\n", sum);

    sum = Serial_pi(n);
    printf("   Single-threaded estimate of pi = %.15f\n", sum);
    printf("   Math library estimate of pi    = %.15f\n", 4.0*atan(1.0));

    free(thread_handles);

    return 0;
}

/*****************************************************************************
 * Function:        Thread_sum
 * Purpose:         Add in the terms computed by the thread running this
 * In args:         rank
 * Return:          ignored(NULL)
 * Globals in:      n, thread_count
 * Global in/out:   sum
 *****************************************************************************/
void* Thread_sum(void* rank)
{
    long my_rank = (long)rank;
    long long my_n = n / thread_count;
    long long my_first_i = my_n * my_rank;
    long long my_last_i = my_first_i + my_n;

    double factor;
    if (my_first_i % 2 == 0)
        factor = 1.0;
    else
        factor = -1.0;

    for (long long i = my_first_i; i < my_last_i; i++, factor = -factor)
        sum += factor/(2*i + 1);

    return NULL;
}

void Get_args(int argc, char* argv[])
{
    int ok = 1;
    if (argc == 3) {
        thread_count = strtol(argv[1], NULL, 10);
        if (thread_count < 0 || thread_count > MAX_THREADS)
            ok = 0;
        
        n = strtoll(argv[2], NULL, 10);
        if (n <= 0)
            ok = 0;
    }
    else
        ok = 0;
    
    if (ok == 0) {
        fprintf(stderr, "Usage: %s <number of threads> <n>\n", argv[0]);
        fprintf(stderr, "   n is the number of terms and should be >= 1\n");
        fprintf(stderr, "   n should be evenly divisible by the number of threads\n");
        exit(0);
    }
}

double Serial_pi(long long n)
{
    double sum = 0.0;
    double factor = 1.0;

    for (long long i = 0; i < n; i++, factor = -factor)
        sum += factor / (2*i + 1);
    
    return 4.0 * sum;
}