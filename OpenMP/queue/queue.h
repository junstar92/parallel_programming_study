/*****************************************************************************
 * File:        queue.h
 * Purpose:     Header file for queue.c which implements a queue of messages or
 *              pairs of ints (source + contents) as a linked list.
 *****************************************************************************/
#ifndef _QUEUE_H_
#define _QUEUE_H_
#ifdef USE_OMP_LOCK
#include <omp.h>
#endif

typedef struct queue_node_s {
    int src;
    int msg;
    struct queue_node_s* next_p;
} QNode;

typedef struct queue_s {
#ifdef USE_OMP_LOCK
    omp_lock_t lock;
#endif
    int enqueued;
    int dequeued;
    QNode* front_p;
    QNode* tail_p;
} Queue;

Queue* Allocate_queue(void);
void Free_queue(Queue* q);
void Print_queue(Queue* q);
void Enqueue(Queue* q, int src, int msg);
int Dequeue(Queue* q, int* src, int* msg);
int Search(Queue* q, int msg, int* src);

#endif