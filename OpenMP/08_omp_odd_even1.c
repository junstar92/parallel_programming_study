/*****************************************************************************
 * File:        08_omp_odd_even1.c
 * Purpose:     Use odd-even transposition sort to sort a list of ints
 * Compile:     gcc -Wall -fopenmp -o 08_omp_odd_even1 08_omp_odd_even1.c
 * Run:         ./08_omp_odd_even1 <number of threads> <n> <g|i>
 *                  n:  number of elements in list
 *                 'g': generate list using a random number generator
 *                 'i': user input list
 * 
 * Input:       list (optional)
 * Output:      elapsed time for sort
 *****************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#ifdef DEBUG
const int RMAX = 100;
#else
const int RMAX = 10000000;
#endif

int thread_count;

void Usage(char* prog_name);
void Get_args(int argc, char* argv[], int* p_n, char* p_g_i);
void Generate_list(int a[], int n);
void Read_list(int a[], int n);
void Print_list(int a[], int n, char* title);
void Odd_even(int a[], int n);

int main(int argc, char* argv[])
{
    int n, *a;
    char g_i;
    
    Get_args(argc, argv, &n, &g_i);
    a = (int*)malloc(n * sizeof(int));
    if (g_i == 'g') {
        Generate_list(a, n);
#ifdef DEBUG
        Print_list(a, n, "Before sort");
#endif
    }
    else {
        Read_list(a, n);
    }

    double start, finish;
    start = omp_get_wtime();
    Odd_even(a, n);
    finish = omp_get_wtime();

#ifdef DEBUG
    Print_list(a, n, "After sort");
#endif

    printf("Elapsed time = %f seconds\n", finish - start);
    
    free(a);
    return 0;
}

/*****************************************************************************
 * Function:        Usage
 * Purpose:         Print a message indicating how program should be started
 *                  and terminate.
 *****************************************************************************/
void Usage(char* prog_name) {
    fprintf(stderr, "Usage:   %s <thread count> <n> <g|i>\n", prog_name);
    fprintf(stderr, "   n:   number of elements in list\n");
    fprintf(stderr, "  'g':  generate list using a random number generator\n");
    fprintf(stderr, "  'i':  user input list\n");
}

/*****************************************************************************
 * Function:        Get_args
 * Purpose:         Get and check command line arguments
 * In args:         argc, argv
 * Out args:        p_n, p_g_i
 *****************************************************************************/
void Get_args(int argc, char* argv[], int* p_n, char* p_g_i)
{
    if (argc != 4) {
        Usage(argv[0]);
        exit(0);
    }

    thread_count = strtol(argv[1], NULL, 10);
    *p_n = strtol(argv[2], NULL, 10);
    *p_g_i = argv[3][0];

    if (*p_n <= 0 || (*p_g_i != 'g' && *p_g_i != 'i')) {
        Usage(argv[0]);
        exit(0);
    }
}

/*****************************************************************************
 * Function:        Generate_list
 * Purpose:         Use random number generator to generate list elements
 * In args:         n
 * Out args:        a
 *****************************************************************************/
void Generate_list(int a[], int n)
{
    srand(n);
    for (int i = 0; i < n; i++)
        a[i] = rand() % RMAX;
}

/*****************************************************************************
 * Function:        Read_list
 * Purpose:         Read elements of list from stdin
 * In args:         n
 * Out args:        a
 *****************************************************************************/
void Read_list(int a[], int n)
{
    printf("Please enter the elements of the list\n");
    for (int i = 0; i < n; i++)
        scanf("%d", &a[i]);
}

/*****************************************************************************
 * Function:        Print_list
 * Purpose:         Print elements in the list
 * In args:         a, n, title
 *****************************************************************************/
void Print_list(int a[], int n, char* title)
{
    printf("%s:\n", title);
    for (int i = 0 ; i < n; i++)
        printf("%d ", a[i]);
    printf("\n\n");
}

/*****************************************************************************
 * Function:        Odd_even
 * Purpose:         Sort list using odd-even transposition sort
 * In args:         n
 * In/out args:     a
 *****************************************************************************/
void Odd_even(int a[], int n)
{
#ifdef DEBUG
    char title[100];
#endif
    int tmp;
    for (int phase = 0; phase < n; phase++) {
        if (phase % 2 == 0) {
#pragma omp parallel for num_threads(thread_count) \
    default(none) shared(a, n) private(tmp)
            for (int i = 1; i < n; i += 2) {
                if (a[i-1] > a[i]) {
                    tmp = a[i-1];
                    a[i-1] = a[i];
                    a[i] = tmp;
                }
            }
        }
        else {
#pragma omp parallel for num_threads(thread_count) \
    default(none) shared(a, n) private(tmp)
            for (int i = 1; i < n-1; i += 2) {
                if (a[i] > a[i+1]) {
                    tmp = a[i+1];
                    a[i+1] = a[i];
                    a[i] = tmp;
                }
            }
        }
#ifdef DEBUG
        sprintf(title, "After phase %d", phase);
        Print_list(a, n , title);
#endif
    }
}