/*****************************************************************************
 * File:        12_mpi_odd_even_sort_unsafe.c
 * Purpose:     Implement parallel odd-even tranposition sort of a list of ints.
 *              This program is unsafe in large global_n because of 
 *              using MPI_Send and MPI_Recv.
 * Compile:     mpicc -Wall -o 12_mpi_odd_even_sort_unsafe 12_mpi_odd_even_sort_unsafe.c
 * Run:         mpiexec -n <p> 12_mpi_odd_even_sort_unsafe <global_n> <g|i>
 *                  - p: the number of processes
 *                  - global_n: the number of elements in list
 *                  - g: generate list using a random number generator
 *                  - i: user input list
 * 
 * Input:       list (optional)
 * Output:      sorted list
 *****************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <string.h>
//#define DEBUG

const int RMAX = 100;

void Usage(char* prog_name);
void Get_args(int argc, char* argv[], int* p_global_n, int* p_local_n, char* p_g_i, int my_rank, int comm_sz, MPI_Comm comm);
void Generate_list(int local_list[], int local_n, int my_rank);
void Read_list(int local_list[], int local_n, int my_rank, int comm_sz, MPI_Comm comm);
void Print_list(int local_list[], int local_n, int my_rank);
void Print_local_lists(int local_list[], int local_n, int my_rank, int comm_sz, MPI_Comm comm);
void Print_global_list(int local_list[], int local_n, int my_rank, int comm_sz, MPI_Comm comm);

int Compare(const void* p_a, const void* p_b);
void Sort(int local_list[], int local_n, int my_rank, int comm_sz, MPI_Comm comm);
void Merge_low(int local_list[], int recv_list[], int temp_list[], int local_n);
void Merge_high(int local_list[], int recv_list[], int temp_list[], int local_n);
void Odd_even_iter(int local_list[], int recv_list[], int temp_list[],
            int local_n, int phase, int even_partner, int odd_partner,
            int my_rank, int comm_sz, MPI_Comm comm);

int main(int argc, char* argv[])
{
    int global_n, local_n;
    char g_i;
    int *local_list;
    int my_rank, comm_sz;
    MPI_Comm comm;

    MPI_Init(&argc, &argv);
    comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &comm_sz);
    MPI_Comm_rank(comm, &my_rank);

    Get_args(argc, argv, &global_n, &local_n, &g_i, my_rank, comm_sz, comm);
    
    local_list = (int*)malloc(local_n * sizeof(int));
    if (g_i == 'g') {
        Generate_list(local_list, local_n, my_rank);
        Print_local_lists(local_list, local_n, my_rank, comm_sz, comm);
    }
    else {
        Read_list(local_list, local_n, my_rank, comm_sz, comm);
    }

#ifdef DEBUG
    if (my_rank == 0) {
        printf("Before Sort\n");
    }
    Print_global_list(local_list, local_n, my_rank, comm_sz, comm);
#endif

    double start, finish, local_elpased, elpased;
    MPI_Barrier(comm);
    start = MPI_Wtime();
    Sort(local_list, local_n, my_rank, comm_sz, comm);
    finish = MPI_Wtime();
    local_elpased = finish - start;
    MPI_Reduce(&local_elpased, &elpased, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

    Print_global_list(local_list, local_n, my_rank, comm_sz, comm);
    if (my_rank == 0)
        printf("Elapsed time = %f seconds\n", elpased);

    free(local_list);

    MPI_Finalize();

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
    fprintf(stderr, "Usage:  mpirun -np <p> %s <global_n> <g|i>\n", prog_name);
    fprintf(stderr, "   p:   the number of processes\n");
    fprintf(stderr, "   global_n:   number of elements in global list\n");
    fprintf(stderr, "   g:   generate list using a random number generator\n");
    fprintf(stderr, "   i:   user input list\n");
}

/*****************************************************************************
 * Function:        Get_args
 * Purpose:         Get and check command list arguments
 * Arguments:
 *  - argc:         The number of arguments
 *  - argv:         The arguments in command list
 *  - p_global_n:   global number of elements of list
 *  - p_local_n:    local number of elements of list
 *  - p_g_i:        Indicator if random generation or user input
 *  - my_rank:      calling process' rank in comm
 *  - comm_sz:      number of processed in comm
 *  - comm:         communicator containing processes calling Get_args
 *****************************************************************************/
void Get_args(int argc, char* argv[], int* p_global_n, int* p_local_n, char* p_g_i,
            int my_rank, int comm_sz, MPI_Comm comm)
{
    if (my_rank == 0) {
        if (argc != 3) {
            Usage(argv[0]);
            *p_global_n = -1;
        }
        else {
            *p_g_i = argv[2][0];
            if (*p_g_i != 'g' && *p_g_i != 'i') {
                Usage(argv[0]);
                *p_global_n = -1;
            }
            else {
                *p_global_n = atoi(argv[1]);
                if (*p_global_n % comm_sz != 0) {
                    Usage(argv[0]);
                    *p_global_n = -1;
                }
            }
        }
    }

    MPI_Bcast(p_g_i, 1, MPI_CHAR, 0, comm);
    MPI_Bcast(p_global_n, 1, MPI_INT, 0, comm);

    if (*p_global_n <= 0) {
        MPI_Finalize();
        exit(-1);
    }

    *p_local_n = *p_global_n / comm_sz;

#ifdef DEBUG
    printf("Proc %d > g_i = %c, global_n = %d, local_n = %d\n", my_rank, *p_g_i, *p_global_n, *p_local_n);
#endif
}

/*****************************************************************************
 * Function:        Generate_list
 * Purpose:         Fill list with random ints
 * Arguments:
 *  - local_list:   local list
 *  - local_n:      local number of elements of list
 *  - my_rank:      calling process' rank in comm
 *****************************************************************************/
void Generate_list(int local_list[], int local_n, int my_rank)
{
    srand(my_rank+1);
    for (int i = 0; i < local_n; i++)
        local_list[i] = rand() % RMAX;
}

/*****************************************************************************
 * Function:        Read_list
 * Purpose:         user input from stdin on process 0 and scatters it to 
 *                  the other processes.
 * Arguments:
 *  - local_list:   The local list
 *  - local_n:      local number of elements of list
 *  - my_rank:      calling process' rank in comm
 *  - comm_sz:      number of processed in comm
 *  - comm:         communicator containing processes calling Read_list
 *****************************************************************************/
void Read_list(int local_list[], int local_n,
        int my_rank, int comm_sz, MPI_Comm comm)
{
    int* temp;

    if (my_rank == 0) {
        temp = (int*)malloc(comm_sz * local_n * sizeof(int));
        printf("Enter the elements of the list\n");
        for (int i = 0; i < comm_sz * local_n; i++)
            scanf("%d", &temp[i]);
    }

    MPI_Scatter(temp, local_n, MPI_INT,
        local_list, local_n, MPI_INT, 0, comm);
    
    if (my_rank == 0)
        free(temp);
}

/*****************************************************************************
 * Function:        Print_list
 * Purpose:         Print list. It only is called by process 0
 * Arguments:
 *  - local_list:   The local list
 *  - local_n:      local number of elements of list
 *  - my_rank:      calling process' rank in comm
 *****************************************************************************/
void Print_list(int local_list[], int local_n, int my_rank)
{
    printf("%d: ", my_rank);
    for (int i = 0; i < local_n; i++)
        printf("%d ", local_list[i]);
    printf("\n");
}

/*****************************************************************************
 * Function:        Print_local_lists
 * Purpose:         Print each process' list contents
 * Arguments:
 *  - local_list:   The local list
 *  - local_n:      local number of elements of list
 *  - my_rank:      calling process' rank in comm
 *  - comm_sz:      number of processed in comm
 *  - comm:         communicator containing processes calling Print_local_lists
 *****************************************************************************/
void Print_local_lists(int local_list[], int local_n,
        int my_rank, int comm_sz, MPI_Comm comm)
{
    int* temp_list;
    MPI_Status status;

    if (my_rank == 0) {
        temp_list = (int*)malloc(local_n * sizeof(int));
        Print_list(local_list, local_n, my_rank);
        for (int q = 1; q < comm_sz; q++) {
            MPI_Recv(temp_list, local_n, MPI_INT, q, 0, comm, &status);
            Print_list(temp_list, local_n, q);
        }
    }
    else {
        MPI_Send(local_list, local_n, MPI_INT, 0, 0, comm);
    }
}

/*****************************************************************************
 * Function:        Print_global_list
 * Purpose:         Print the contents of t he global list
 * Arguments:
 *  - local_list:   The local list
 *  - local_n:      local number of elements of list
 *  - my_rank:      calling process' rank in comm
 *  - comm_sz:      number of processed in comm
 *  - comm:         communicator containing processes calling Print_global_list
 *****************************************************************************/
void Print_global_list(int local_list[], int local_n,
        int my_rank, int comm_sz, MPI_Comm comm)
{
    int* global_list;

    if (my_rank == 0) {
        int global_n = comm_sz * local_n;
        global_list = (int*)malloc(global_n * sizeof(int));
        MPI_Gather(local_list, local_n, MPI_INT,
            global_list, local_n, MPI_INT, 0, comm);

        printf("Global list:\n");
        for (int i = 0 ; i < global_n; i++)
            printf("%d ", global_list[i]);
        printf("\n\n");

        free(global_list);
    }
    else {
        MPI_Gather(local_list, local_n, MPI_INT,
            global_list, local_n, MPI_INT, 0, comm);
    }
}

/*****************************************************************************
 * Function:        Compare
 * Purpose:         Compare 2 ints, return -1, 0, or 1
 *                  -1: the first int is less than the second
 *                   0: the first int is equal the second
 *                   1: the first int is greater than the second
 * Arguments:
 *  - p_a:          the first int
 *  - p_b:          the second int
 *****************************************************************************/
int Compare(const void* p_a, const void* p_b)
{
    int a = *((int*)p_a);
    int b = *((int*)p_b);

    if (a < b)
        return -1;
    else if (a == b)
        return 0;
    else
        return 1;
}

/*****************************************************************************
 * Function:        Sort
 * Purpose:         Sort local list, use odd-even sort to sort global list.
 * Arguments:
 *  - local_list:   The local list
 *  - local_n:      local number of elements of list
 *  - my_rank:      calling process' rank in comm
 *  - comm_sz:      number of processed in comm
 *  - comm:         communicator containing processes calling Sort
 *****************************************************************************/
void Sort(int local_list[], int local_n, int my_rank, int comm_sz, MPI_Comm comm)
{
    int* recv_list, *temp_list;
    int even_partner, odd_partner;

    recv_list = (int*)malloc(local_n * sizeof(int));
    temp_list = (int*)malloc(local_n * sizeof(int));

    if (my_rank % 2 != 0) {
        even_partner = my_rank - 1;
        odd_partner = my_rank + 1;
        if (odd_partner == comm_sz)
            odd_partner = MPI_PROC_NULL;
    }
    else {
        even_partner = my_rank + 1;
        if (even_partner == comm_sz)
            even_partner = MPI_PROC_NULL;
        odd_partner = my_rank - 1;
    }

    /* Sort local list using built-in quick sort */
    qsort(local_list, local_n, sizeof(int), Compare);

    for (int phase = 0; phase < comm_sz; phase++)
        Odd_even_iter(local_list, recv_list, temp_list, local_n, phase,
                even_partner, odd_partner, my_rank, comm_sz, comm);
    
    free(recv_list);
    free(temp_list);
}

/*****************************************************************************
 * Function:        Merge_low
 * Purpose:         Merge the smallest local_n elements in local_lists and
 *                  recv_list into temp_list, Then copy temp_list back into
 *                  local_list
 * Arguments:
 *  - local_list:   The local list
 *  - recv_list:    The list recieved from partner process
 *  - temp_list:    The temporary list to save the results
 *  - local_n:      local number of elements of list
 *****************************************************************************/
void Merge_low(int local_list[], int recv_list[], int temp_list[], int local_n)
{
    int local_i = 0, recv_i = 0, temp_i = 0;

    while (temp_i < local_n) {
        if (local_list[local_i] <= recv_list[recv_i]) {
            temp_list[temp_i] = local_list[local_i];
            local_i++; temp_i++;
        }
        else {
            temp_list[temp_i] = recv_list[recv_i];
            recv_i++; temp_i++;
        }
    }

    memcpy(local_list, temp_list, local_n * sizeof(int));
}

/*****************************************************************************
 * Function:        Merge_high
 * Purpose:         Merge the largest local_n elements in local_lists and
 *                  recv_list into temp_list, Then copy temp_list back into
 *                  local_list
 * Arguments:
 *  - local_list:   The local list
 *  - recv_list:    The list recieved from partner process
 *  - temp_list:    The temporary list to save the results
 *  - local_n:      local number of elements of list
 *****************************************************************************/
void Merge_high(int local_list[], int recv_list[], int temp_list[], int local_n)
{
    int local_i, recv_i, temp_i;
    local_i = recv_i = temp_i = local_n - 1;

    while (temp_i >= 0) {
        if (local_list[local_i] >= recv_list[recv_i]) {
            temp_list[temp_i] = local_list[local_i];
            local_i--; temp_i--;
        }
        else {
            temp_list[temp_i] = recv_list[recv_i];
            recv_i--; temp_i--;
        }
    }

    memcpy(local_list, temp_list, local_n * sizeof(int));
}

/*****************************************************************************
 * Function:        Odd_even_iter
 * Purpose:         Print each process' list contents
 * Arguments:
 *  - local_list:   The local list
 *  - recv_list:    The list recieved from partner process
 *  - temp_list:    The temporary list to save the results
 *  - local_n:      local number of elements of list
 *  - phase:        phase of sorting process
 *  - even_partner: even partner process rank of calling process
 *  - odd_partner:  odd partner process rank of calling process
 *  - my_rank:      calling process' rank in comm
 *  - comm_sz:      number of processed in comm
 *  - comm:         communicator containing processes calling Odd_even_iter
 *****************************************************************************/
void Odd_even_iter(int local_list[], int recv_list[], int temp_list[],
            int local_n, int phase, int even_partner, int odd_partner,
            int my_rank, int comm_sz, MPI_Comm comm)
{
    MPI_Status status;

    if (phase % 2 == 0) {
        /* even phase */
        if (even_partner >= 0) {
            MPI_Send(local_list, local_n, MPI_INT, even_partner, 0, comm);
            MPI_Recv(recv_list, local_n, MPI_INT, even_partner, 0, comm, &status);

            if (my_rank % 2 != 0)
                Merge_high(local_list, recv_list, temp_list, local_n);
            else
                Merge_low(local_list, recv_list, temp_list, local_n);
        }
    }
    else {
        /* odd phase */
        if (odd_partner >= 0) {
            MPI_Send(local_list, local_n, MPI_INT, odd_partner, 0, comm);
            MPI_Recv(recv_list, local_n, MPI_INT, odd_partner, 0, comm, &status);

            if (my_rank % 2 != 0)
                Merge_low(local_list, recv_list, temp_list, local_n);
            else
                Merge_high(local_list, recv_list, temp_list, local_n);
        }
    }
}