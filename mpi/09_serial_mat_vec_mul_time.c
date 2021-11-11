/*****************************************************************************
 * File:        09_serial_mat_vec_mul_time.c
 * Purpose:     Implement serial matrix-vector multiplication using
 *              one-dimensional arrays to store the vectors and the matrix and
 *              estimate running time.
 * Compile:     gcc -Wall -o 09_serial_mat_vec_mul_time 09_serial_mat_vec_mul_time.c
 * Run:         ./09_serial_mat_vec_mul_time
 * 
 * Input:       Dimension of the matrix (m = number of rows,
 *                                       n = number of columns)
 *              m x n matrix A
 *              n-dimensional vector x
 * Output:      Vector y = Ax
 *              Elapsed time
 *****************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

/* The argument now should be a double (not a pointer to a double) */
#define GET_TIME(now) \
{ \
    struct timeval tv; \
    gettimeofday(&tv, NULL); \
    now = tv.tv_sec + tv.tv_usec/1000000.0; \
}

const int RMAX = 10000000;

void Get_dims(int* p_m, int* p_n);
void Allocate_arrays(double** pp_A, double** pp_x, double** pp_y, int m, int n);
void Get_matrix(double mat[], int m, int n, char mat_name[]);
void Get_vector(double vec[], int n, char vec_name[]);
void Print_matrix(double mat[], int m, int n, char title[]);
void Print_vector(double vec[], int n, char title[]);
void Mat_vec_mul(double A[], double x[], double y[], int m, int n);

int main(void)
{
    double *A, *x, *y;
    int m, n;

    Get_dims(&m, &n);
    Allocate_arrays(&A, &x, &y, m, n);

    Get_matrix(A, m, n, "A");
    //Print_matrix(A, m, n, "A");
    Get_vector(x, n, "x");
    //Print_vector(x, n, "x");

    double start, finish;
    GET_TIME(start);
    Mat_vec_mul(A, x, y, m, n);
    GET_TIME(finish);
    Print_vector(y, m, "y");

    printf("Elapsed time = %f seconds\n", finish - start);

    free(A);
    free(x);
    free(y);

    return 0;
}

/*****************************************************************************
 * Function:        Get_dims
 * Purpose:         Get the dimensions of the matrix and the vectors from stdin
 * Arguments:
 *  - p_m:          global number of rows of A and components of y
 *  - p_n:          global number of cols of A and components of x
 *****************************************************************************/
void Get_dims(
    int*        p_m         /* out */,
    int*        p_n         /* out */)
{
    printf("Enter the number of rows\n");
    scanf("%d", p_m);
    printf("Enter the number of columns\n");
    scanf("%d", p_n);

    if (*p_m <= 0 || *p_n <= 0) {
        fprintf(stderr, "Dimension of matrix and vector should be > 0");
        exit(0);
    }
}

/*****************************************************************************
 * Function:        Allocate_arrays
 * Purpose:         Allocate storage for A, x, and y
 * Arguments:
 *  - pp_A:         storage for matrix (m/comm_sz x n)
 *  - pp_x:         storage for x (n/comm_sz components)
 *  - pp_y:         storage for y (m/comm_sz components)
 *  - m:            the number of rows of A and components of y
 *  - n:            the number of cols of A and components of x
 *****************************************************************************/
void Allocate_arrays(
    double**    pp_A        /* out */,
    double**    pp_x        /* out */,
    double**    pp_y        /* out */,
    int         m           /* in  */,
    int         n           /* in  */)
{
    *pp_A = (double*)malloc(m * n * sizeof(double));
    *pp_x = (double*)malloc(n * sizeof(double));
    *pp_y = (double*)malloc(m * sizeof(double));

    if ((*pp_A == NULL) || (*pp_x == NULL) || (*pp_y == NULL)) {
        fprintf(stderr, "Can't allocate arrays");
        exit(-1);
    }
}

/*****************************************************************************
 * Function:        Get_matrix
 * Purpose:         Get matrix from rand function
 * Arguments:
 *  - mat:          the matrix read
 *  - m:            the number of rows of matrix
 *  - n:            the number of cols of matrix
 *  - mat_name:     description of matrix (e.g., "A")
 *****************************************************************************/
void Get_matrix(
    double      mat[]       /* out */,
    int         m           /* in  */,
    int         n           /* in  */,
    char        mat_name[]  /* in  */)
{
    srand(m);
    printf("Get the matrix %s from rand function\n", mat_name);
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            mat[i*n + j] = (rand() % RMAX) / (RMAX/10.0);
}

/*****************************************************************************
 * Function:        Get_vector
 * Purpose:         Get a vector from rand function
 * Arguments:
 *  - vec:          vector read
 *  - n:            size of vectors
 *  - vec_name:     name of vector being read (e.g., "x")
 *****************************************************************************/
void Get_vector(
    double      vec[]       /* out */,
    int         n           /* in  */,
    char        vec_name[]  /* in  */)
{
    srand(n);
    printf("Get the vector %s from rand function\n", vec_name);
    for (int i = 0; i < n; i++) {
        vec[i] = (rand() % RMAX) / (RMAX/10.0);
    }
}

/*****************************************************************************
 * Function:        Print_matrix
 * Purpose:         Print a matrix to stdout
 * Arguments:
 *  - mat:          matrix
 *  - m:            the number of rows
 *  - n:            the number of cols
 *  - title:        name of matrix
 *****************************************************************************/
void Print_matrix(
    double      mat[]       /* in */,
    int         m           /* in */,
    int         n           /* in */,
    char        title[]     /* in */)
{
    printf("\nThe matrix %s\n", title);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++)
            printf("%f ", mat[i*n + j]);
        printf("\n");
    }
    printf("\n");
}

/*****************************************************************************
 * Function:        Print_vector
 * Purpose:         Print a vector to stdout
 * Arguments:
 *  - local_vec:    the vector
 *  - n:            the number of components
 *  - title:        name of vector
 *****************************************************************************/
void Print_vector(
    double      vec[]       /* in */,
    int         n           /* in */,
    char        title[]     /* in */)
{
    printf("\nThe vector %s\n", title);
    for (int i = 0 ; i < n; i++) {
        printf("%f ", vec[i]);
    }
    printf("\n");
}

/*****************************************************************************
 * Function:        Mat_vec_mul
 * Purpose:         Implement serial matrix-vector multiplication
 * Arguments:
 *  - A:            matrix A
 *  - x:            vector x
 *  - y:            vector to save result of multiplation
 *                  y = Ax
 *  - m:            the number of rows
 *  - n:            the number of columns
 *****************************************************************************/
void Mat_vec_mul(
    double      A[]         /* in  */,
    double      x[]         /* in  */,
    double      y[]         /* out */,
    int         m           /* in  */,
    int         n           /* in  */)
{
    for (int i = 0; i < m; i++) {
        y[i] = 0.0;
        for (int j = 0; j < n; j++)
            y[i] += A[i*n + j] * x[j];
    }
}