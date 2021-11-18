/*****************************************************************************
 * File:        12_pth_tokenize.c
 * Purpose:     Try to use threads to tokenize text input. Illustrate problems
 *              with function that isn't thread-safe.
 * 
 *              * This program deinitely has problems.
 * 
 * Compile:     gcc -Wall -o 12_pth_tokenize 12_pth_tokenize.c -pthread
 * Run:         ./12_pth_tokenize <number of threads>
 * 
 * Input:       Lines of text
 * Output:      For each line of input:
 *                the line read by the program, and the tokens identified by
 *                strtok
 *****************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <semaphore.h>

const int MAX = 1000;

long thread_count;
sem_t* sems;

void Usage(char* prog_name);
void *Tokenize(void* rank); /* thread function */

int main(int argc, char* argv[])
{
    if (argc != 2)
        Usage(argv[0]);
    thread_count = atoi(argv[1]);
    
    pthread_t* thread_handles = (pthread_t*)malloc(thread_count * sizeof(pthread_t));
    sems = (sem_t*)malloc(thread_count * sizeof(sem_t));
    // sems[0] should be unlocked, the others should be locked
    sem_init(&sems[0], 0, 1);
    for (long thread = 1; thread < thread_count; thread++)
        sem_init(&sems[thread], 0, 0);
    
    printf("Enter text\n");
    for (long thread = 0; thread < thread_count; thread++)
        pthread_create(&thread_handles[thread], NULL, Tokenize, (void*)thread);
    
    for (long thread = 0; thread < thread_count; thread++)
        pthread_join(thread_handles[thread], NULL);
    
    for (long thread = 0; thread < thread_count; thread++)
        sem_destroy(&sems[thread]);
    
    free(sems);
    free(thread_handles);
    return 0;
}

/*****************************************************************************
 * Function:        Usage
 * Purpose:         Print command line for function and terminate
 * In args:         prog_name
 *****************************************************************************/
void Usage(char* prog_name)
{
    fprintf(stderr, "Usage: %s <number of threads>\n", prog_name);
    exit(0);
}

/*****************************************************************************
 * Function:        Tokenize
 * Purpose:         Tokenize lines of input
 * In args:         rank
 * Global var:      thread_count, sems
 * Return:          ignored(NULL)
 *****************************************************************************/
void* Tokenize(void* rank)
{
    long my_rank = (long)rank;
    int count;
    int next = (my_rank + 1) % thread_count;
    char* fg_rv;
    char my_line[MAX];
    char* my_string;

    /* Force sequential reading of the input */
    sem_wait(&sems[my_rank]);
    fg_rv = fgets(my_line, MAX, stdin);
    sem_post(&sems[next]);

    while (fg_rv != NULL) {
        printf("Thread %ld > my_line = %s", my_rank, my_line);

        count = 0;
        my_string = strtok(my_line, " \t\n");
        while (my_string != NULL) {
            count++;
            printf("Thread %ld > string %d = %s\n", my_rank, count, my_string);
            my_string = strtok(NULL, " \t\n");
        }

        //if (my_line != NULL)
            //printf("Thread %ld > After tokenizing, my_line = %s\n", my_rank, my_line);
        
        sem_wait(&sems[my_rank]);
        fg_rv = fgets(my_line, MAX, stdin);
        sem_post(&sems[next]);
    }

    return NULL;
}