/*****************************************************************************
 * File:        09_pth_sem_barrier.c
 * Purpose:     Use semaphore barriers to synchronize threads
 * 
 * Compile:     gcc -Wall -o 09_pth_sem_barrier 09_pth_sem_barrier.c -pthread
 *              [-DDEBUG]
 * Run:         ./09_pth_sem_barrier <number of threads>
 * 
 * Input:       none
 * Output:      Time for BARRIER_COUNT barriers
 * 
 * Note:        Verbose output can be enabled with the compile flag -DDEBUG
 *****************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <semaphore.h>
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
long counter;
sem_t barrier_sems[BARRIER_COUNT];
sem_t count_sem;

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
    for (int i = 0; i < BARRIER_COUNT; i++)
        sem_init(&barrier_sems[i], 0, 0);
    sem_init(&count_sem, 0, 1);

    double start, finish;
    GET_TIME(start);
    for (long thread = 0; thread < thread_count; thread++)
        pthread_create(&thread_handles[thread], NULL, Thread_work, (void*)thread);
    for (long thread = 0; thread < thread_count; thread++)
        pthread_join(thread_handles[thread], NULL);
    GET_TIME(finish);

    printf("Elapsed time = %f seconds\n", finish - start);

    sem_destroy(&count_sem);
    for (int i = 0; i < BARRIER_COUNT; i++)
        sem_destroy(&barrier_sems[i]);
    free(thread_handles);

    return 0;
}

/*****************************************************************************
 * Function:        Thread_work
 * Purpose:         Run BARRIER_COUNT barriers
 * In args:         rank
 * Global var:      thread_count, count, barrier_sems, count_sem
 * Return:          ignored(NULL)
 *****************************************************************************/
void* Thread_work(void* rank)
{
#ifdef DEBUG
    long my_rank = (long)rank;
#endif

    for (int i = 0; i < BARRIER_COUNT; i++) {
        sem_wait(&count_sem);
        if (counter == thread_count - 1) {
            counter = 0;
            sem_post(&count_sem);
            for (int j = 0; j < thread_count - 1; j++)
                sem_post(&barrier_sems[i]);
        }
        else {
            counter++;
            sem_post(&count_sem);
            sem_wait(&barrier_sems[i]);
        }
#ifdef DEBUG
        if (my_rank == 0) {
            printf("All threads completed barrier %d\n", i);
            fflush(stdout);
        }
#endif
    }

    return NULL;
}