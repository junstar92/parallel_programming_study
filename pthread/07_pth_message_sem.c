/*****************************************************************************
 * File:        07_pth_message_sem.c
 * Purpose:     Illustrate a synchronization problem with pthreads:
 *              create some threads, each of which creates and sends it to
 *              another thread, by copying it into that thread's buffer.
 *              This version uses semaphores to solve the synchronization problem
 * 
 * Compile:     gcc -Wall -o 07_pth_message_sem 07_pth_message_sem.c -pthread [-lm]
 * Run:         ./07_pth_message_sem <number of threads>
 * 
 * Input:       none
 * Output:      message from each thread
 *****************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <semaphore.h>

const int MAX_THREADS = 1024;
const int MSG_MAX = 100;

/* Global variables */
long  thread_count;
char** messages;
sem_t* semaphores;

void* Send_message(void* rank); /* Thread function */

int main(int argc, char* argv[])
{
    pthread_t* thread_handles;

    if (argc != 2) {
        fprintf(stderr, "Usage: %s <number of threads>\n", argv[0]);
        exit(0);
    }

    thread_count = strtol(argv[1], NULL, 10);
    if (thread_count <= 0 || thread_count > MAX_THREADS) {
        fprintf(stderr, "The number of threads should be > 0 and < %d\n", MAX_THREADS);
        exit(0);
    }

    thread_handles = (pthread_t*)malloc(thread_count * sizeof(pthread_t));
    messages = (char**)malloc(thread_count * sizeof(char*));
    semaphores = (sem_t*)malloc(thread_count * sizeof(sem_t));

    for (long thread = 0; thread < thread_count; thread++) {
        messages[thread] = NULL;
        sem_init(&semaphores[thread], 0, 0);
    }
    for (long thread = 0; thread < thread_count; thread++)
        pthread_create(&thread_handles[thread], NULL, Send_message, (void*)thread);
    for (long thread = 0; thread < thread_count; thread++)
        pthread_join(thread_handles[thread], NULL);
    
    for (long thread = 0; thread < thread_count; thread++) {
        free(messages[thread]);
        sem_destroy(&semaphores[thread]);
    }
    free(messages);
    free(semaphores);
    free(thread_handles);

    return 0;
}

/*****************************************************************************
 * Function:        Send_message
 * Purpose:         Create a message and send it by copying it into
 *                  global messages array. Receive a message and print it.
 * In args:         rank
 * Global in:       thread_count
 * Global in/out:   messages, semaphores
 * Return:          ignored(NULL)
 * Node:            The my_msg buffer is freed in main function
 *****************************************************************************/
void* Send_message(void* rank)
{
    long my_rank = (long)rank;
    long dest = (my_rank + 1) % thread_count;
    char* my_msg = (char*)malloc(MSG_MAX * sizeof(char));

    sprintf(my_msg, "Hello to %ld from %ld", dest, my_rank);
    messages[dest] = my_msg;
    sem_post(&semaphores[dest]); // increase semaphores[dest] by 1 -> 'unlock' the semaphore of dest

    sem_wait(&semaphores[my_rank]); // decrease semaphores[my_rank] by 1 and return -> wait for our semaphore to be unlocked
    printf("Thread %ld > %s\n", my_rank, messages[my_rank]);

    return NULL;
}