/*****************************************************************************
 * File:        00_pth_hello.c
 * Purpose:     Illustrate basic use of threads: create some threads,
 *              each of which prints a mssage.
 * Compile:     gcc -Wall -o 00_pth_hello 00_pth_hello.c -pthread
 * Run:         ./00_pth_hello <thread_count>
 * 
 * Input:       none
 * Output:      message from each thread
 *****************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

const int MAX_THREADS = 64;

/* global variables: accesible to all threads */
int thread_count;

void Usage(char* prog_name);
void* Hello(void* rank);

int main(int argc, char* argv[])
{
    if (argc != 2) {
        Usage(argv[0]);
    }

    /* Get number of threads from command line */
    thread_count = strtol(argv[1], NULL, 10);
    if (thread_count <= 0 || thread_count > MAX_THREADS) {
        Usage(argv[0]);
    }

    pthread_t* thread_handles;
    thread_handles = malloc(thread_count*sizeof(pthread_t));

    for (long thread = 0; thread < thread_count; thread++) {
        pthread_create(&thread_handles[thread], NULL, Hello, (void*)thread);
    }

    printf("Hello from the main thread\n");

    for (long thread = 0; thread < thread_count; thread++) {
        pthread_join(thread_handles[thread], NULL);
    }

    free(thread_handles);

    return 0;
}

void Usage(char* prog_name)
{
    fprintf(stderr, "Usage %s <number of threads>\n", prog_name);
    fprintf(stderr, "0 < number of threads <= %d\n", MAX_THREADS);
    exit(0);
}

void* Hello(void* rank)
{
    long my_rank = (long)rank;

    printf("Hello from thread %ld of %d\n", my_rank, thread_count);

    return NULL;
}