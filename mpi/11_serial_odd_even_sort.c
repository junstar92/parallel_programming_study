/*****************************************************************************
 * File:        11_serial_odd_even_sort.c
 * Purpose:     Use odd-even transposition sort to sort a list of ints
 * Compile:     gcc -Wall -o 11_serial_odd_even_sort 11_serial_odd_even_sort.c
 * Run:         ./11_serial_odd_even_sort <n> <g|i>
 *                  - n: the number of elements in list
 *                  - g: generate list using a random number generator
 *                  - i: user input list
 * 
 * Input:       list (optional)
 * Output:      sorted list
 *****************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define GET_TIME(now) \
{ \
    struct timeval tv; \
    gettimeofday(&tv, NULL); \
    now = tv.tv_sec + tv.tv_usec/1000000.0; \
}

const int RMAX = 100;

void Usage(char* prog_name);
void Get_args(int argc, char* argv[], int* p_n, char* p_g_i);
void Generate_list(int list[], int n);
void Read_list(int list[], int n);
void Print_list(int list[], int n, char* title);
void Odd_even_sort(int list[], int n);

int main(int argc, char* argv[])
{
    int n;
    char g_i;
    int* a;

    Get_args(argc, argv, &n, &g_i);

    a = (int*)malloc(n * sizeof(int));
    if (g_i == 'g') {
        Generate_list(a, n);
        Print_list(a, n, "Before sort");
    }
    else {
        Read_list(a, n);
    }

    double start, finish;
    GET_TIME(start);
    Odd_even_sort(a, n);
    GET_TIME(finish);

    Print_list(a, n, "After sort");
    printf("Elapsed time = %f seconds\n", finish - start);

    free(a);

    return 0;
}

/*****************************************************************************
 * Function:        Usage
 * Purpose:         Summary of how to run program
 * Arguments:
 *  - prog_name:    Program Name
 *****************************************************************************/
void Usage(char* prog_name)
{
    fprintf(stderr, "Usage:   %s <n> <g|i>\n", prog_name);
    fprintf(stderr, "   n:   number of elements in list\n");
    fprintf(stderr, "   g:   generate list using a random number generator\n");
    fprintf(stderr, "   i:   user input list\n");
}

/*****************************************************************************
 * Function:        Get_args
 * Purpose:         Get and check command list arguments
 * Arguments:
 *  - argc:         The number of arguments
 *  - argv:         The arguments in command list
 *  - p_n:          The pointer to n
 *  - p_g_i:        The pointer to g_i 
 *****************************************************************************/
void Get_args(int argc, char* argv[], int* p_n, char* p_g_i)
{
    if (argc != 3) {
        Usage(argv[0]);
        exit(0);
    }

    *p_n = atoi(argv[1]);
    *p_g_i = argv[2][0];

    if (*p_n <= 0 || (*p_g_i != 'g' && *p_g_i != 'i')) {
        Usage(argv[0]);
        exit(0);
    }
}

/*****************************************************************************
 * Function:        Generate_list
 * Purpose:         Generate list elements using random number generator
 * Arguments:
 *  - list:         list
 *  - n:            the number of list
 *****************************************************************************/
void Generate_list(int list[], int n)
{
    srand(n);

    for (int i = 0; i < n; i++)
        list[i] = rand() % RMAX;
}

/*****************************************************************************
 * Function:        Read_list
 * Purpose:         Read elements of list from stdin
 * Arguments:
 *  - list:         list
 *  - n:            the number of list
 *****************************************************************************/
void Read_list(int list[], int n)
{
    printf("Enter the elements of the list\n");
    for (int i = 0; i < n; i++)
        scanf("%d", &list[i]);
}

/*****************************************************************************
 * Function:        Print_list
 * Purpose:         Print the elements in the list to stdout
 * Arguments:
 *  - list:         list
 *  - n:            the number of list
 *  - title:        title
 *****************************************************************************/
void Print_list(int list[], int n, char* title)
{
    printf("%s:\n", title);
    for (int i = 0 ; i < n; i++)
        printf("%d ", list[i]);
    printf("\n\n");
}

/*****************************************************************************
 * Function:        Odd_even_sort
 * Purpose:         Sort list using odd-even transposition sort
 * Arguments:
 *  - list:         list
 *  - n:            the number of list
 *  - title:        title
 *****************************************************************************/
void Odd_even_sort(
    int list[]  /* in/out */,
    int n       /* in     */)
{
    int tmp;
    for (int phase = 0; phase < n; phase++) {
        if (phase % 2 == 0) {
            /* even phase */
            for (int i = 1; i < n; i += 2) {
                if (list[i-1] > list[i]) {
                    tmp = list[i];
                    list[i] = list[i-1];
                    list[i-1] = tmp;
                }
            }
        }
        else {
            /* odd phase */
            for (int i = 1; i < n-1; i += 2) {
                if (list[i] > list[i+1]) {
                    tmp = list[i];
                    list[i] = list[i+1];
                    list[i+1] = tmp;
                }
            }
        }
    }
}