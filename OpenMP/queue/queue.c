/*****************************************************************************
 * File:        queue.c
 * Purpose:     Implement a queue of pairs of ints (msg source and contents)
 *              using a linked list. Operations are Enqueue, Dequeue, Print,
 *              Search, and Free
 * Compile:     gcc -Wall -DUSE_MAIN -o queue queue.c
 * Run:         ./queue
 * 
 * Input:       Operations (first letter of op name) and, when necessary, keys
 * Output:      Prompts for input and results of operations
 *****************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include "queue.h"

#ifdef USE_MAIN
int main(void)
{
    char op;
    int src, msg, not_empty;
    Queue* q = Allocate_queue();

    printf("Op? (e, d, p, s, f, q)\n");
    scanf(" %c", &op);
    while (op != 'q' && op != 'Q') {
        switch (op) {
            case 'e':
            case 'E':
                printf("Src? Msg?\n");
                scanf("%d%d", &src, &msg);
#ifdef USE_OMP_LOCK
                omp_set_lock(&q->lock);
#endif
                Enqueue(q, src, msg);
#ifdef USE_OMP_LOCK
                omp_unset_locak(q->lock);
#endif
                break;
            
            case 'd':
            case 'D':
#ifdef USE_OMP_LOCK
                omp_set_lock(&q->lock);
#endif
                not_empty = Dequeue(q, &src, &msg);
#ifdef USE_OMP_LOCK
                omp_unset_lock(&q->lock);
#endif
                if (not_empty) {
                    printf("Dequeue src = %d, msg = %d\n", src, msg);
                }
                else {
                    printf("Queue is empty\n");
                }
                break;
            
            case 's':
            case 'S':
                printf("Msg?\n");
                scanf("%d", &msg);
                if (Search(q, msg, &src)) {
                    printf("Found %d from %d\n", msg, src);
                }
                else {
                    printf("Didn't find %d\n", msg);
                }
                break;
            
            case 'p':
            case 'P':
                Print_queue(q);
                break;
            
            case 'f':
            case 'F':
#ifdef USE_OMP_LOCK
                omp_set_lock(&q->lock);
#endif
                Free_queue(q);
#ifdef USE_OMP_LOCK
                omp_unset_lock(&q->lock);
#endif
                break;
            
            default:
                printf("%c isn't a valid command\n", op);
                printf("Please try again\n");
        }
        printf("Op? (e, d, p, s, f, q)\n");
        scanf(" %c", &op);
    }

    Free_queue(q);
#ifdef USE_OMP_LOCK
    omp_destroy_lock(&q->lock);
#endif
    free(q);

    return 0;
}
#endif

/*****************************************************************************
 * Function:        Allocate_queue
 * Purpose:         Allocate queue structure
 * In args:         none
 * Return:          pointer of queue allocated
 *****************************************************************************/
Queue* Allocate_queue(void)
{
    Queue* q = (Queue*)malloc(sizeof(Queue));
    q->enqueued = q->dequeued = 0;
    q->front_p = q->tail_p = NULL;
#ifdef USE_OMP_LOCK
    omp_init_lock(&q->lock);
#endif

    return q;
}

/*****************************************************************************
 * Function:        Free_queue
 * Purpose:         Free queue
 * In args:         q
 * Return:          none
 *****************************************************************************/
void Free_queue(Queue* q)
{
    QNode* curr = q->front_p;
    QNode* temp;

    while (curr != NULL) {
        temp = curr;
        curr = curr->next_p;
        free(temp);
    }
    q->enqueued = q->dequeued = 0;
    q->front_p = q->tail_p = NULL;
}

/*****************************************************************************
 * Function:        Print_queue
 * Purpose:         Print queue details to stdout
 * In args:         q
 * Return:          none
 *****************************************************************************/
void Print_queue(Queue* q)
{
    QNode* curr = q->front_p;

    printf("queue = \n");
    while (curr != NULL) {
        printf("   src = %d, msg = %d\n", curr->src, curr->msg);
        curr = curr->next_p;
    }
    printf("enqueued = %d, dequeued = %d\n", q->enqueued, q->dequeued);
    printf("\n");
}

/*****************************************************************************
 * Function:        Enqueue
 * Purpose:         Push queue node to front of queue
 * In args:         q, src, msg
 * Return:          none
 *****************************************************************************/
void Enqueue(Queue* q, int src, int msg)
{
    QNode* node = (QNode*)malloc(sizeof(QNode));
    node->src = src;
    node->msg = msg;
    node->next_p = NULL;

    if (q->tail_p == NULL) {
        q->front_p = node;
        q->tail_p = node;
    }
    else {
        q->tail_p->next_p = node;
        q->tail_p = node;
    }
    q->enqueued++;
}

/*****************************************************************************
 * Function:        Dequeue
 * Purpose:         Pop front of queue
 * In args:         q, src, msg
 * Return:          0 or 1,
 *                      0 - fail
 *                      1 - success
 *****************************************************************************/
int Dequeue(Queue* q, int* src, int* msg)
{
    QNode* temp;

    if (q->front_p == NULL)
        return 0;
    
    *src = q->front_p->src;
    *msg = q->front_p->msg;
    temp = q->front_p;
    
    if (q->front_p == q->tail_p) { /* One node in list */
        q->front_p = q->tail_p = NULL;
    }
    else {
        q->front_p = temp->next_p;
    }

    free(temp);
    q->dequeued++;

    return 1;
}

/*****************************************************************************
 * Function:        Search
 * Purpose:         Search source has the msg
 * In args:         q, src, msg
 * Return:          0 or 1,
 *                      0 - fail
 *                      1 - success
 *****************************************************************************/
int Search(Queue* q, int msg, int *src)
{
    QNode* curr = q->front_p;

    while (curr != NULL) {
        if (curr->msg == msg) {
            *src = curr->src;
            return 1;
        }
        else {
            curr = curr->next_p;
        }
    }

    return 0;
}