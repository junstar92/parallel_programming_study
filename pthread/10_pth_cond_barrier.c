/*****************************************************************************
 * File:        10_pth_cond_barrier.c
 * Purpose:     Use condition wait barriers to synchronize threads
 * 
 * Compile:     gcc -Wall -o 10_pth_cond_barrier 10_pth_cond_barrier.c -pthread
 *              [-DDEBUG]
 * Run:         ./10_pth_cond_barrier <number of threads>
 * 
 * Input:       none
 * Output:      Time for BARRIER_COUNT barriers
 * 
 * Note:        Verbose output can be enabled with the compile flag -DDEBUG
 *****************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <sys/time.h>

#define GET_TIME(now)                           \
    {                                           \
        struct timeval t;                       \
        gettimeofday(&t, NULL);                 \
        now = t.tv_sec + t.tv_usec / 1000000.0; \
    }
//#define DEBUG
#define BARRIER_COUNT 100

/* Global variables */
long thread_count, barrier_thread_count;
pthread_mutex_t barrier_mutex;
pthread_cond_t ok_to_proceed;

void* Thread_work(void* rank);

int main(int argc, char* argv[])
{
    pthread_t* thread_handles;

    if (argc != 2) {
        fprintf(stderr, "Usage: %s <number of threads>\n", argv[0]);
        exit(0);
    }

    thread_count = strtol(argv[1], NULL, 10);
    if (thread_count <= 0) {
        fprintf(stderr, "The number of threads should be > 0\n");
        exit(0);
    }

    thread_handles = (pthread_t*)malloc(thread_count * sizeof(pthread_t));
    pthread_mutex_init(&barrier_mutex, NULL);
    pthread_cond_init(&ok_to_proceed, NULL);

    double start, finish;
    GET_TIME(start);
    for (long thread = 0; thread < thread_count; thread++)
        pthread_create(&thread_handles[thread], NULL, Thread_work, (void*)thread);
    for (long thread = 0; thread < thread_count; thread++)
        pthread_join(thread_handles[thread], NULL);
    GET_TIME(finish);

    printf("Elapsed time = %f seconds\n", finish - start);

    pthread_mutex_destroy(&barrier_mutex);
    pthread_cond_destroy(&ok_to_proceed);
    free(thread_handles);

    return 0;
}

/*****************************************************************************
 * Function:        Thread_work
 * Purpose:         Run BARRIER_COUNT barriers
 * In args:         rank
 * Global var:      thread_count, barrier_thread_count, barrier_mutex
 * Return:          ignored(NULL)
 *****************************************************************************/
void* Thread_work(void* rank)
{
#ifdef DEBUG
    long my_rank = (long)rank;
#endif

    for (int i = 0; i < BARRIER_COUNT; i++) {
        pthread_mutex_lock(&barrier_mutex);
        barrier_thread_count++;

        if (barrier_thread_count == thread_count) {
            barrier_thread_count = 0;
#ifdef DEBUG
            printf("Thread %ld > Signalling other threads in barrier %d\n", my_rank, i);
            fflush(stdout);
#endif
            pthread_cond_broadcast(&ok_to_proceed);
        }
        else {
            // Wait unlocks mutex and puts thread to sleep.
            //    Put wait in while loop in case some other
            // event awakens thread.
            while (pthread_cond_wait(&ok_to_proceed, &barrier_mutex) != 0);
            // Mutex is relocked at this point.
#ifdef DEBUG
            printf("Thread %ld > Awakened in barrier %d\n", my_rank, i);
#endif
        }
        pthread_mutex_unlock(&barrier_mutex);
#ifdef DEBUG
        if (my_rank == 0) {
            printf("All threads completed barrier %d\n", i);
            fflush(stdout);
        }
#endif
    }

    return NULL;
}