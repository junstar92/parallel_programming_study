/*****************************************************************************
 * File:        ocv_mat_mul.c
 * Purpose:     Compute a matrix-matrix product by using OpenCV library.
 * Compile:     g++ -Wall -o ocv_mat_mul ocv_mat_mul.c $(pkg-config opencv4 --libs --cflags)
 * Run:         ./ocv_mat_mul <m> <n> <k>
 *                  <m> : the rows of matrix A
 *                  <n> : the columns of matrix A and the rows of matrix B
 *                  <k> : the columns of matrix B
 * 
 * Input:       A, B
 * Output:      
 *              C: the product matrix, C = AB
 *              Elapsed time each multiplication and average elapsed time of
 *              100 multiplications
 *****************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <opencv2/highgui.hpp>
#include <sys/time.h>

#define GET_TIME(now) { \
    struct timeval t; \
    gettimeofday(&t, NULL); \
    now = t.tv_sec + t.tv_usec/1000000.0; \
}

const int RMAX = 1000000;
#ifdef DEBUG
const int NCOUNT = 1; // number of multiplication
#else
const int NCOUNT = 100; // number of multiplication
#endif

void Get_args(int argc, char* argv[], int* m, int* n, int* k);
void Usage(char* prog_name);
void Generate_matrix(double mat[], int m, int n);
void Print_matrix(double mat[], int m, int n, char* title);

int main(int argc, char* argv[])
{
    int m, n, k;
    Get_args(argc, argv, &m, &n, &k);

    double *A, *B, *C;
    A = (double*)malloc(m * n * sizeof(double));
    B = (double*)malloc(n * k * sizeof(double));
    C = (double*)malloc(m * k * sizeof(double));

    Generate_matrix(A, m, n);
    Generate_matrix(B, n, k);
#ifdef DEBUG
    Print_matrix(A, m, n, "A");
    Print_matrix(B, m, n, "B");
#endif


    double start, finish, avg_elapsed = 0.0;
    for (int count = 0; count < NCOUNT; count++) {
        GET_TIME(start);
        cv::Mat cvA(m, n, CV_64FC1, A);
        cv::Mat cvB(n, k, CV_64FC1, B);
        cv::Mat cvC = cvA * cvB;
        //cv::gemm(cvA, cvB, 1.0, NULL, 0, cvC);
        //C = reinterpret_cast<double*>(cvC.data);
        GET_TIME(finish);

        printf("[%3d] Elapsed time = %.6f seconds\n", count+1, finish-start);
        avg_elapsed += (finish - start) / NCOUNT;
    }
    
#ifdef DEBUG
    printf("The product is\n");
    cv::print(C);
    printf("\n\n");
#endif

    printf("Average elapsed time = %.6f seconds\n", avg_elapsed);

    free(A);
    free(B);
    free(C);

    return 0;
}

/*****************************************************************************
 * Function:        Get_args
 * Purpose:         Get and check command list arguments
 * In args:         argc, argv
 * Out args:        m, n, k
 *****************************************************************************/
void Get_args(int argc, char* argv[], int* m, int* n, int* k)
{
    if (argc != 4)
        Usage(argv[0]);
    
    *m = strtol(argv[1], NULL, 10);
    *n = strtol(argv[2], NULL, 10);
    *k = strtol(argv[3], NULL, 10);
    if (*m <= 0 || *n <= 0 || *k <= 0)
        Usage(argv[0]);
}

/*****************************************************************************
 * Function:        Usage
 * Purpose:         Print a message indicating how program should be started
 *                  and terminate.
 * In arg:          prog_name
 *****************************************************************************/
void Usage(char* prog_name)
{
    fprintf(stderr, "Usage: %s <m> <n> <k>\n", prog_name);
    exit(0);
}

/*****************************************************************************
 * Function:        Generate_matrix
 * Purpose:         Generate matrix entries by using the random number generator
 * In args:         m, n
 * Out arg:         mat
 *****************************************************************************/
void Generate_matrix(double mat[], int m, int n)
{
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            mat[i*n + j] = (rand() % RMAX) / (RMAX / 10.0);
}

/*****************************************************************************
 * Function:        Print_matrix
 * Purpose:         Print the matrix
 * In args:         mat, m, n, title
 *****************************************************************************/
void Print_matrix(double mat[], int m, int n, char* title)
{
    printf("%s\n", title);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++)
            printf("%f ", mat[i*n + j]);
        printf("\n");
    }
}