/*****************************************************************************
 * File:        06_pth_message.c
 * Purpose:     Illustrate a synchronization problem with pthreads:
 *              create some threads, each of which creates and sends it to
 *              another thread, by copying it into that thread's buffer.
 * 
 * Compile:     gcc -Wall -o 06_pth_message 06_pth_message.c -pthread [-lm]
 * Run:         ./06_pth_message <number of threads>
 * 
 * Input:       none
 * Output:      message from each thread
 *****************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

const int MAX_THREADS = 1024;
const int MSG_MAX = 100;

/* Global variables */
long thread_count;
char** messages;

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

    for (long thread = 0; thread < thread_count; thread++)
        messages[thread] = NULL;
    for (long thread = 0; thread < thread_count; thread++)
        pthread_create(&thread_handles[thread], NULL, Send_message, (void*)thread);
    for (long thread = 0; thread < thread_count; thread++)
        pthread_join(thread_handles[thread], NULL);
    
    for (long thread = 0; thread < thread_count; thread++)
        free(messages[thread]);
    free(messages);

    free(thread_handles);

    return 0;
}

/*****************************************************************************
 * Function:        Send_message
 * Purpose:         Create a message and send it by copying it into
 *                  global messages array. Receive a message and print it.
 * In args:         rank
 * Global in:       thread_count
 * Global in/out:   messages
 * Return:          ignored(NULL)
 * Node:            The my_msg buffer is freed in main function
 *****************************************************************************/
void* Send_message(void* rank)
{
    long my_rank = (long)rank;
    long dest = (my_rank + 1) % thread_count;
    long src = (my_rank + thread_count - 1) % thread_count;
    char* my_msg = (char*)malloc(MSG_MAX * sizeof(char));

    sprintf(my_msg, "Hello to %ld from %ld", dest, my_rank);
    messages[dest] = my_msg;

    if (messages[my_rank] != NULL)
        printf("Thread %ld > %s\n", my_rank, messages[my_rank]);
    else
        printf("Thread %ld > No message from %ld\n", my_rank, src);

    return NULL;
}