/*****************************************************************************
 * File:        11_pth_posix_barrier.c
 * Purpose:     Use POSIX barrier to synchronize threads
 * 
 * Compile:     gcc -Wall -o 11_pth_posix_barrier 11_pth_posix_barrier.c -pthread
 *              [-DDEBUG]
 * Run:         ./11_pth_posix_barrier <number of threads>
 * 
 * Input:       none
 * Output:      Time for BARRIER_COUNT barriers
 * 
 * Node:        Verbose output can be enabled with the compile flag -DDEBUG
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

#define BARRIER_COUNT 100

/* Global variables */
long thread_count;
pthread_barrier_t barrier;

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
    pthread_barrier_init(&barrier, NULL, thread_count);

    double start, finish;
    GET_TIME(start);
    for (long thread = 0; thread < thread_count; thread++)
        pthread_create(&thread_handles[thread], NULL, Thread_work, (void*)thread);
    for (long thread = 0; thread < thread_count; thread++)
        pthread_join(thread_handles[thread], NULL);
    GET_TIME(finish);

    printf("Elapsed time = %f seconds\n", finish - start);

    pthread_barrier_destroy(&barrier);
    free(thread_handles);

    return 0;
}

/*****************************************************************************
 * Function:        Thread_work
 * Purpose:         Run BARRIER_COUNT barriers
 * In args:         rank
 * Global var:      thread_count, barrier
 * Return:          ignored(NULL)
 *****************************************************************************/
void* Thread_work(void* rank)
{
#ifdef DEBUG
    long my_rank = (long)rank;
#endif

    for (int i = 0; i < BARRIER_COUNT; i++) {
        pthread_barrier_wait(&barrier);
#ifdef DEBUG
        if (my_rank == 0) {
            printf("All threads completed barrier %d\n", i);
            fflush(stdout);
        }
#endif
    }

    return NULL;
}