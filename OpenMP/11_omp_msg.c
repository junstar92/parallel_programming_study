/*****************************************************************************
 * File:        11_omp_msg.c
 * Purpose:     Simulate message-passing using openMP. Uses critical(or lock)
 *              and atomic directives to protect critical sections.
 * Compile:     gcc -Wall -fopenmp -o 11_omp_msg 11_omp_msg.c ./queue/queue.c
 *              [-DDEBUG] [-DUSE_OMP_LOCK]
 *              (needs queue.h)
 * Run:         ./11_omp_msg <number of threads> <number of messages each 
 *                  thread sends>
 * 
 * Input:       None
 * Output:      Source, destination and contents of each message received.
 * 
 * Note:        If add '-DUSE_OMP_LOCK' option, uses locks to control access to
 *              the message queues instead of omp critical
 *****************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "queue/queue.h"

const int MAX_MSG = 10000;

void Usage(char* prog_name);
void Send_msg(Queue* msg_queues[], int my_rank, int thread_count, int msg_number);
void Try_receive(Queue* q, int my_rank);
int Done(Queue* q, int done_sending, int thread_count);

int main(int argc, char* argv[])
{
    if (argc != 3)
        Usage(argv[0]);
    int thread_count = strtol(argv[1], NULL, 10);
    int send_max = strtol(argv[2], NULL, 10);
    if (thread_count <= 0 || send_max < 0)
        Usage(argv[0]);

    Queue** msg_queues = (Queue**)malloc(thread_count * sizeof(Queue*));
    int done_sending = 0;
#pragma omp parallel num_threads(thread_count) \
    default(none) shared(thread_count, send_max, msg_queues, done_sending)
    {
        int my_rank = omp_get_thread_num();
        srand(my_rank);
        msg_queues[my_rank] = Allocate_queue();

#pragma omp barrier /*  Don't let any threads send messages 
                        until all queue are contructed      */
        for (int msg_number = 0; msg_number < send_max; msg_number++) {
            Send_msg(msg_queues, my_rank, thread_count, msg_number);
            Try_receive(msg_queues[my_rank], my_rank);
        }
#pragma omp atomic
        done_sending++;
#ifdef DEBUG
        printf("Thread %d > done sending\n", my_rank);
#endif

        while (!Done(msg_queues[my_rank], done_sending, thread_count))
            Try_receive(msg_queues[my_rank], my_rank);
        
        /*  My queue is empty, and everyone is done sending
            So my queue won't be accessed again, and it's OK to free it */
        Free_queue(msg_queues[my_rank]);
        free(msg_queues[my_rank]);
    } /* omp parallel */

    free(msg_queues);
    return 0;
}

/*****************************************************************************
 * Function:        Usage
 * Purpose:         Print a message indicating how program should be started
 *                  and terminate.
 *****************************************************************************/
void Usage(char* prog_name)
{
    fprintf(stderr, "Usage: %s <number of threads> <number of messages>\n", prog_name);
    fprintf(stderr, "   number of messages = number sent by each thread\n");
    exit(0);
}

/*****************************************************************************
 * Function:        Send_msg
 * Purpose:         Create a message and push the message to message queue of
 *                  random thread
 * In args:         msg_queues[], my_rank, thread_count, msg_number
 *****************************************************************************/
void Send_msg(Queue* msg_queues[], int my_rank, int thread_count, int msg_number)
{
    int msg = rand() % MAX_MSG;
    int dest = rand() % thread_count;
#pragma omp critical
    Enqueue(msg_queues[dest], my_rank, msg);
#ifdef DEBUG
    printf("Thread %d > sent %d to %d\n", my_rank, msg, dest);
#endif
}

/*****************************************************************************
 * Function:        Try_receive
 * Purpose:         Try to receive message from queue and print information
 * In args:         q, my_rank
 *****************************************************************************/
void Try_receive(Queue* q, int my_rank)
{
    int src, msg;
    int q_size = q->enqueued - q->dequeued;

    if (q_size == 0)
        return;
    else if (q_size == 1)
#pragma omp critical
        Dequeue(q, &src, &msg);
    else
        Dequeue(q, &src, &msg);
    
    printf("Thread %d > received %d from %d\n", my_rank, msg, src);
}

/*****************************************************************************
 * Function:        Done
 * Purpose:         Check if sending message is done and queue is empty
 * In args:         q, done_sending, thread_count
 *****************************************************************************/
int Done(Queue* q, int done_sending, int thread_count)
{
    int q_size = q->enqueued - q->dequeued;

    if (q_size == 0 && done_sending == thread_count)
        return 1;
    else
        return 0;
}