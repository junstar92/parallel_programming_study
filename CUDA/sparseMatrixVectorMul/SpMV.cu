/*****************************************************************************
 * File:        SpMV.cu
 * Description: Calculate AX + Y by several methods.
 *              A : N x N sparse matrix
 *              X : N vector
 *              Y : N vector
 *              Result is saved in vector Y.
 *              
 * Compile:     nvcc -o SpMV SpMV.cu -I..
 * Run:         ./SpMV
 * Argument:
 *      "--n=<N>"           : Specify the number of N (default: 4096)
 *      "--threads=<N>"     : Specify the number of threads per block (default: 512)
 *      "--seed=<N>"        : Specify random seed value (default: 0)
 *      "--ratio=<N>"       : Specify ratio of 0 element in Matrix (0 ~ 100) (default: 20)
 *****************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <random>
#include <cuda_runtime.h>

#include <common/common.h>
#include <common/common_string.h>

int initSparseMatrix(float* mat, int rows, int cols, int ratio);
void initVector(float* vec, int num_elem);
void printMatrix(float* data, int rows, int cols);
void printMatrix(int* mat, int rows, int cols);
void printVector(float* vec, int num_elem);
void printVector(int* vec, int num_elem);

// Function for CSR
void decomposeCSR(float* mat, int rows, int cols, float *data, int *col_index, int *row_ptr);
void SpMV_CPU(float* data, int rows, int cols, float* x, float* y);
void SpMV_CSR_seq(int num_rows, float* data, int* col_index, int* row_ptr, float* x, float* y);
__global__ void SpMV_CSR(int num_rows, float* data, int* col_index, int* row_ptr, float* x, float* y);

// Function for ELL
int getMaxNumElemInRows(int *row_ptr, int num_rows);
void decomposeELL_FromCSR(float** ell_data, int** ell_col_index, int num_rows, float* csr_data, int* csr_col_index, int* csr_row_ptr, int maxNumInRows);
__global__ void SpMV_ELL(int num_rows, float* data, int* col_index, int num_elem, float* x, float* y);

// Function for Hybrid ELL-COO
int decomposeHybridELLandCOO_FromCSR(float** ell_data, int** ell_col_index, float** coo_data, int** coo_col_index, int ** coo_row_index, int limitNum,
                            int num_rows, float* csr_data, int* csr_col_index, int* csr_row_ptr);
void SpMV_COO_seq(float* data, int* col_index, int* row_index, int num_elem, float* x, float* y);
__global__ void SpMV_COO(float* data, int* col_index, int* row_index, int num_elem, float* x, float* y);

int main(int argc, char** argv)
{
    printf("[Sparse Matrix - Vector(SpMV) Multiplication]\n\n");
    int N = 4096;
    int threads = 512;
    int seed = 0;
    int ratio = 20;

    if (checkCmdLineFlag(argc, (const char **)argv, "n")) {
        N = getCmdLineArgumentInt(argc, (const char **)argv, "n");
    }
    if (checkCmdLineFlag(argc, (const char **)argv, "threads")) {
        threads = getCmdLineArgumentInt(argc, (const char **)argv, "threads");
    }
    if (checkCmdLineFlag(argc, (const char **)argv, "seed")) {
        seed = getCmdLineArgumentInt(argc, (const char **)argv, "seed");
    }
    if (checkCmdLineFlag(argc, (const char **)argv, "ratio")) {
        ratio = getCmdLineArgumentInt(argc, (const char **)argv, "ratio");
    }

    printf("N : %d\n", N);
    printf("The number of elements in Sparse Matrix: %d\n", N*N);
    printf("The ratio of 0 in Sparse matrix: %d %%\n\n", ratio);

    srand(seed);
    
    int blocks = (N + threads - 1) / threads;
    printf("The number of threads per block: %d\n", threads);
    printf("The number of blocks: %d\n", blocks);

    int dev = 0;
    CUDA_CHECK(cudaSetDevice(dev));

    unsigned int mat_bytes = N*N*sizeof(float);
    unsigned int vec_bytes = N*sizeof(float);

    printf("Origin Sparse Matrix Storage: %d MBytes (%d bytes)\n\n", mat_bytes / (1024*1024), mat_bytes);

    float *h_SparseMat_A, *h_vec_X, *h_vec_Y;
    int num_nonzero;

    // allocate host memory
    h_SparseMat_A = (float*)malloc(mat_bytes);
    h_vec_X = (float*)malloc(vec_bytes);
    h_vec_Y = (float*)malloc(vec_bytes);

    // init value
    num_nonzero = initSparseMatrix(h_SparseMat_A, N, N, ratio);
    initVector(h_vec_X, N);
    (void)memset(h_vec_Y, 0, vec_bytes);
    //initVector(h_vec_Y, N);

#ifdef DEBUG
    printf("\nMatrix A\n");
    printMatrix(h_SparseMat_A, N, N);
    printf("\nVector X\n");
    printVector(h_vec_X, N);
#endif

    // array for CSR
    float *h_data;
    int *col_index, *row_ptr;

    // decompose sparse matrix for CSR Format
    printf("Get Array for CSR Format...\n");
    h_data = (float*)malloc(num_nonzero*sizeof(float));
    col_index = (int*)malloc(num_nonzero*sizeof(int));
    row_ptr = (int*)malloc((N+1)*sizeof(int));
    decomposeCSR(h_SparseMat_A, N, N, h_data, col_index, row_ptr);
#ifdef DEBUG
    printf("data: ");
    printVector(h_data, num_nonzero);
    printf("col_index: ");
    printVector(col_index, num_nonzero);
    printf("row_ptr: ");
    printVector(row_ptr, N+1);

    SpMV_CPU(h_SparseMat_A, N, N, h_vec_X, h_vec_Y);
    printf("Result of vector Y: ");
    printVector(h_vec_Y, N);
#endif
    double start, finish, total_time = 0.f;
    int nIter = 50;

    printf("The Number of non-zero elements: %d\n", num_nonzero);
    printf("Total Storage for CSR Format: %ld MBytes (%ld bytes)\n",
            (num_nonzero*sizeof(float) + num_nonzero*sizeof(int) + (N+1)*sizeof(int)) / (1024*1024), (num_nonzero*sizeof(float) + num_nonzero*sizeof(int) + (N+1)*sizeof(int)));
    printf("\n[Sequential SpMV/CSR computation...]\n");
    for (int i = 0; i < nIter; i++) {
        (void)memset(h_vec_Y, 0, vec_bytes);

        GET_TIME(start);
        SpMV_CSR_seq(N, h_data, col_index, row_ptr, h_vec_X, h_vec_Y);
        GET_TIME(finish);
        total_time += (finish - start);
    }
    printf("Elapsed Time: %.6f ms\n", (total_time / (double)nIter)*1000);

#ifdef DEBUG
    printf("Result of Sequential SpMV computation\n");
    printf("Vector Y\n");
    printVector(h_vec_Y, N);
#endif
    float *d_vec_X, *d_vec_Y;

    // allocate device memory
    CUDA_CHECK(cudaMalloc((void**)&d_vec_X, vec_bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_vec_Y, vec_bytes));

    // copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_vec_X, h_vec_X, vec_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_vec_Y, 0, vec_bytes));


    /**************** SpMV/CSR Kernel Launch ****************/
    printf("\n[SpMV/CSR computation...]\n");
    float *d_data;
    int *d_col_index, *d_row_ptr;
    CUDA_CHECK(cudaMalloc((void**)&d_data, num_nonzero*sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_col_index, num_nonzero*sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_row_ptr, (N+1)*sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_data, h_data, num_nonzero*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_index, col_index, num_nonzero*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_row_ptr, row_ptr, (N+1)*sizeof(int), cudaMemcpyHostToDevice));

    total_time = 0.f;
    GET_TIME(start);
    decomposeCSR(h_SparseMat_A, N, N, h_data, col_index, row_ptr);
    GET_TIME(finish);
    total_time += (finish - start);
    for (int i = 0; i < nIter; i++) {
        (void)memset(h_vec_Y, 0, vec_bytes);
        CUDA_CHECK(cudaMemset(d_vec_Y, 0, vec_bytes));

        GET_TIME(start);
        SpMV_CSR<<<blocks, threads>>>(N, d_data, d_col_index, d_row_ptr, d_vec_X, d_vec_Y);
        CUDA_CHECK(cudaMemcpy(h_vec_Y, d_vec_Y, vec_bytes, cudaMemcpyDeviceToHost));
        GET_TIME(finish);
        total_time += (finish - start);
    }
    printf("Elapsed Time: %.6f ms\n", (total_time / (double)nIter)*1000);

#ifdef DEBUG
    printf("Result of Sequential SpMV computation\n");
    printf("Vector Y\n");
    printVector(h_vec_Y, N);
#endif


    /**************** SpMV/ELL Kernel Launch ****************/
    printf("\n[SpMV/ELL computation...]\n");
    float *h_data_forELL = NULL;
    int *h_col_index_forELL = NULL;
    float *d_data_forELL = NULL;
    int *d_col_index_forELL = NULL;

    int maxElemInData = 0;
    total_time = 0.f;

    GET_TIME(start);
    decomposeCSR(h_SparseMat_A, N, N, h_data, col_index, row_ptr);
    maxElemInData = getMaxNumElemInRows(row_ptr, N+1);
    decomposeELL_FromCSR(&h_data_forELL, &h_col_index_forELL, N, h_data, col_index, row_ptr, maxElemInData);
    GET_TIME(finish);
    total_time += (finish-start);

    CUDA_CHECK(cudaMalloc((void**)&d_data_forELL, N*maxElemInData*sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_col_index_forELL, N*maxElemInData*sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_data_forELL, h_data_forELL, N*maxElemInData*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_index_forELL, h_col_index_forELL, N*maxElemInData*sizeof(int), cudaMemcpyHostToDevice));

    printf("Total Storage for ELL Format: %ld MBytes (%ld bytes)\n",
        (N*maxElemInData*sizeof(float) + N*maxElemInData*sizeof(int)) / (1024*1024), (N*maxElemInData*sizeof(float) + N*maxElemInData*sizeof(int)));
    printf("The maximum number of rows in data: %d\n", maxElemInData);

    for (int i = 0; i < nIter; i++) {
        (void)memset(h_vec_Y, 0, vec_bytes);
        CUDA_CHECK(cudaMemset(d_vec_Y, 0, vec_bytes));

        GET_TIME(start);
        SpMV_ELL<<<blocks, threads>>>(N, d_data_forELL, d_col_index_forELL, maxElemInData, d_vec_X, d_vec_Y);
        CUDA_CHECK(cudaMemcpy(h_vec_Y, d_vec_Y, vec_bytes, cudaMemcpyDeviceToHost));
        GET_TIME(finish);
        total_time += (finish - start);
    }
    printf("Elapsed Time: %.6f ms\n", (total_time / (double)nIter)*1000);

#ifdef DEBUG
    printf("Result of Sequential SpMV computation\n");
    printf("Vector Y\n");
    printVector(h_vec_Y, N);
#endif

    free(h_data_forELL);
    free(h_col_index_forELL);
    /**************** Hybrid SpMV/ELL-COO Kernel Launch ****************/
    printf("\n[SpMV/ELL-COO(Hybrid) computation...]\n");
    float* h_data_forCOO = NULL;
    int* h_col_index_forCOO = NULL;
    int* h_row_index_forCOO = NULL;
    float* d_data_forCOO = NULL;
    int* d_col_index_forCOO = NULL;
    int* d_row_index_forCOO = NULL;

    int limitNum = N / 8;
    total_time = 0.f;

    GET_TIME(start);
    decomposeCSR(h_SparseMat_A, N, N, h_data, col_index, row_ptr);
    maxElemInData = getMaxNumElemInRows(row_ptr, N+1);
    int coo_data_cnt = decomposeHybridELLandCOO_FromCSR(&h_data_forELL, &h_col_index_forELL, &h_data_forCOO, &h_col_index_forCOO, &h_row_index_forCOO,
                                    limitNum, N, h_data, col_index, row_ptr);
    GET_TIME(finish);
    total_time += (finish-start);

    CUDA_CHECK(cudaMalloc((void**)&d_data_forELL, N*limitNum*sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_col_index_forELL, N*limitNum*sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_data_forCOO, coo_data_cnt*sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_col_index_forCOO, coo_data_cnt*sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_row_index_forCOO, coo_data_cnt*sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_data_forELL, h_data_forELL, N*limitNum*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_index_forELL, h_col_index_forELL, N*limitNum*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_data_forCOO, h_data_forCOO, coo_data_cnt*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_index_forCOO, h_col_index_forCOO, coo_data_cnt*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_row_index_forCOO, h_row_index_forCOO, coo_data_cnt*sizeof(int), cudaMemcpyHostToDevice));

    printf("Total Storage for ELL Format: %ld MBytes (%ld bytes)\n",
        (N*limitNum*sizeof(float) + N*limitNum*sizeof(int) + 3*coo_data_cnt*sizeof(int)) / (1024*1024), (N*limitNum*sizeof(float) + N*limitNum*sizeof(int) + 3*coo_data_cnt*sizeof(int)));
    printf("The maximum number of rows in ELL data: %d\n", limitNum);
    printf("The number of COO data: %d\n", coo_data_cnt);

    for (int i = 0; i < nIter; i++) {
        (void)memset(h_vec_Y, 0, vec_bytes);
        CUDA_CHECK(cudaMemset(d_vec_Y, 0, vec_bytes));

        GET_TIME(start);
        SpMV_ELL<<<blocks, threads>>>(N, d_data_forELL, d_col_index_forELL, limitNum, d_vec_X, d_vec_Y);
        SpMV_COO<<<blocks, threads>>>(d_data_forCOO, d_col_index_forCOO, d_row_index_forCOO, coo_data_cnt, d_vec_X, d_vec_Y);
        CUDA_CHECK(cudaMemcpy(h_vec_Y, d_vec_Y, vec_bytes, cudaMemcpyDeviceToHost));
        //SpMV_COO_seq(h_data_forCOO, h_col_index_forCOO, h_row_index_forCOO, coo_data_cnt, h_vec_X, h_vec_Y);
        GET_TIME(finish);
        total_time += (finish - start);
    }
    printf("Elapsed Time: %.6f ms\n", (total_time / (double)nIter)*1000);

#ifdef DEBUG
    printf("Result of Sequential SpMV computation\n");
    printf("Vector Y\n");
    printVector(h_vec_Y, N);
#endif

    /**************** End of launching Kernel ****************/

    // free memory
    free(h_SparseMat_A);
    free(h_vec_X);
    free(h_vec_Y);
    CUDA_CHECK(cudaFree(d_vec_X));
    CUDA_CHECK(cudaFree(d_vec_Y));

    free(h_data);
    free(col_index);
    free(row_ptr);
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_col_index));
    CUDA_CHECK(cudaFree(d_row_ptr));

    free(h_data_forELL);
    free(h_col_index_forELL);
    free(h_data_forCOO);
    free(h_col_index_forCOO);
    free(h_row_index_forCOO);
    CUDA_CHECK(cudaFree(d_data_forELL));
    CUDA_CHECK(cudaFree(d_col_index_forELL));

    printf("\n[Done]\n\n");

    return 0;
}

int initSparseMatrix(float* mat, int rows, int cols, int ratio)
{
    std::default_random_engine generator;
    std::uniform_int_distribution<int> dist(0, 100);
    int num_nonzero = 0;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (dist(generator) < ratio) {
                mat[i*cols + j] = rand() / (float)RAND_MAX;
                num_nonzero++;
            }
            else {
                mat[i*cols + j] = 0.f;
            }
        }
    }

    return num_nonzero;
}

void initVector(float* vec, int num_elem)
{
    for (int i = 0; i < num_elem; i++)
        vec[i] = rand() / (float)RAND_MAX;
}

void printMatrix(float* mat, int rows, int cols)
{
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%.5f ", mat[i*cols + j]);
        }
        printf("\n");
    }
}

void printMatrix(int* mat, int rows, int cols)
{
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%d ", mat[i*cols + j]);
        }
        printf("\n");
    }
}

void printVector(float* vec, int num_elem)
{
    for (int i = 0; i < num_elem; i++) {
        printf("%.5f ", vec[i]);
    }
    printf("\n");
}

void printVector(int* vec, int num_elem)
{
    for (int i = 0; i < num_elem; i++) {
        printf("%d ", vec[i]);
    }
    printf("\n");
}

void decomposeCSR(float* mat, int rows, int cols, float *data, int *col_index, int *row_ptr)
{
    int idx = 0;
    for (int i = 0; i < rows; i++) {
        row_ptr[i] = idx;
        for (int j = 0; j < cols; j++) {
            if (abs(mat[i*cols + j]) > 1.0e-8) {
                data[idx] = mat[i*cols + j];
                col_index[idx] = j;
                idx++;
            }
        }
    }
    row_ptr[rows] = idx;
}

void SpMV_CPU(float* A, int rows, int cols, float* x, float* y)
{
    for (int row = 0; row < rows; row++) {
        float dot = 0;
        for (int col = 0; col < cols; col++) {
            dot += A[row*cols + col] * x[col];
        }
        y[row] = dot;
    }
}

void SpMV_CSR_seq(int num_rows, float* data, int* col_index, int* row_ptr, float* x, float* y)
{
    for (int row = 0; row < num_rows; row++) {
        float dot = 0;
        int row_start = row_ptr[row];
        int row_end = row_ptr[row+1];

        for (int i = row_start; i < row_end; i++)
            dot += data[i] * x[col_index[i]];
        y[row] += dot;
    }
}

__global__
void SpMV_CSR(int num_rows, float* data, int* col_index, int* row_ptr, float* x, float* y)
{
    int row = blockDim.x*blockIdx.x + threadIdx.x;

    if (row < num_rows) {
        float dot = 0;
        int row_start = row_ptr[row];
        int row_end = row_ptr[row+1];

        for (int i = row_start; i < row_end; i++)
            dot += data[i] * x[col_index[i]];
        y[row] += dot;
    }
}

int getMaxNumElemInRows(int *row_ptr, int num_rows)
{
    int ret = 0;
    for (int i = 1; i < num_rows + 1; i++) {
        int tmp = row_ptr[i] - row_ptr[i-1];
        if (tmp > ret)
            ret = tmp;
    }

    return ret;
}

void decomposeELL_FromCSR(float** ell_data, int** ell_col_index, int num_rows, float* csr_data, int* csr_col_index, int* csr_row_ptr, int maxNumInRows)
{
    float* data_temp = (float*)malloc(num_rows*maxNumInRows*sizeof(float));
    int* col_index_temp = (int*)malloc(num_rows*maxNumInRows*sizeof(int));
    *ell_data = (float*)malloc(num_rows*maxNumInRows*sizeof(float));
    *ell_col_index = (int*)malloc(num_rows*maxNumInRows*sizeof(int));

    if (data_temp == NULL || col_index_temp == NULL || ell_data == NULL || ell_col_index == NULL) {
        fprintf(stderr, "Failed to allocate in %s\n", __FUNCTION__);
        exit(EXIT_FAILURE);
    }

    (void)memset(data_temp, 0, num_rows*maxNumInRows*sizeof(float));
    (void)memset(col_index_temp, 0, num_rows*maxNumInRows*sizeof(int));
    (void)memset(*ell_data, 0, num_rows*maxNumInRows*sizeof(float));
    (void)memset(*ell_col_index, 0, num_rows*maxNumInRows*sizeof(int));

    for (int i = 0; i < num_rows; i++) {
        int row_start = csr_row_ptr[i];
        int row_end = csr_row_ptr[i+1];
        int col_idx = 0;
        for (int j = row_start; j < row_end; j++, col_idx++) {
            data_temp[i*maxNumInRows + col_idx] = csr_data[j];
            col_index_temp[i*maxNumInRows + col_idx] = csr_col_index[j];
        }
    }

    // transposition
    for (int i = 0; i < maxNumInRows; i++) {
        for (int j = 0; j < num_rows; j++) {
            (*ell_data)[i*num_rows + j] = data_temp[j*maxNumInRows + i];
            (*ell_col_index)[i*num_rows + j] = col_index_temp[j*maxNumInRows + i];
        }
    }
#ifdef DEBUG
    printf("new data for ELL\n");
    printMatrix(*ell_data, maxNumInRows, num_rows);
    printf("new col index for ELL\n");
    printMatrix(*ell_col_index, maxNumInRows, num_rows);
#endif

    free(data_temp);
    free(col_index_temp);
}

__global__
void SpMV_ELL(int num_rows, float* data, int* col_index, int num_elem, float* x, float* y)
{
    int row = blockDim.x*blockIdx.x + threadIdx.x;

    if (row < num_rows) {
        float dot = 0;
        for (int i = 0; i < num_elem; i++)
            dot += data[row + i*num_rows] * x[col_index[row + i*num_rows]];
        y[row] += dot;
    }
}

int decomposeHybridELLandCOO_FromCSR(float** ell_data, int** ell_col_index, float** coo_data, int** coo_col_index, int ** coo_row_index, int limitNum,
                            int num_rows, float* csr_data, int* csr_col_index, int* csr_row_ptr)
{
    float* data_temp = (float*)malloc(num_rows*limitNum*sizeof(float));
    int* col_index_temp = (int*)malloc(num_rows*limitNum*sizeof(int));
    *ell_data = (float*)malloc(num_rows*limitNum*sizeof(float));
    *ell_col_index = (int*)malloc(num_rows*limitNum*sizeof(int));

    (void)memset(data_temp, 0, num_rows*limitNum*sizeof(float));
    (void)memset(col_index_temp, 0, num_rows*limitNum*sizeof(int));
    (void)memset(*ell_data, 0, num_rows*limitNum*sizeof(float));
    (void)memset(*ell_col_index, 0, num_rows*limitNum*sizeof(int));

    int coo_data_cnt = 0;
    for (int i = 0; i < num_rows; i++) {
        int numElemInRows = csr_row_ptr[i+1] - csr_row_ptr[i];
        if (numElemInRows > limitNum)
            coo_data_cnt += (numElemInRows - limitNum);
    }

    *coo_data = (float*)malloc(coo_data_cnt*sizeof(float));
    *coo_col_index = (int*)malloc(coo_data_cnt*sizeof(int));
    *coo_row_index = (int*)malloc(coo_data_cnt*sizeof(int));

    if (data_temp == NULL || col_index_temp == NULL || *ell_data == NULL || *ell_col_index == NULL ||
        *coo_data == NULL || *coo_col_index == NULL || *coo_row_index == NULL) {
        fprintf(stderr, "Failed to allocate in %s\n", __FUNCTION__);
        exit(EXIT_FAILURE);
    }

    int coo_data_index = 0;
    for (int i = 0; i < num_rows; i++) {
        int row_start = csr_row_ptr[i];
        int row_end = csr_row_ptr[i+1];
        int col_idx = 0;
        for (int j = row_start; j < row_end; j++, col_idx++) {
            if (col_idx < limitNum) {
                data_temp[i*limitNum + col_idx] = csr_data[j];
                col_index_temp[i*limitNum + col_idx] = csr_col_index[j];
            }
            else {
                (*coo_data)[coo_data_index] = csr_data[j];
                (*coo_col_index)[coo_data_index] = csr_col_index[j];
                (*coo_row_index)[coo_data_index] = i;
                coo_data_index++;
            }
        }
    }

    // transposition
    for (int i = 0; i < limitNum; i++) {
        for (int j = 0; j < num_rows; j++) {
            (*ell_data)[i*num_rows + j] = data_temp[j*limitNum + i];
            (*ell_col_index)[i*num_rows + j] = col_index_temp[j*limitNum + i];
        }
    }

#ifdef DEBUG
    printf("COO data count: %d\n", coo_data_cnt);
    printf("new data for ELL\n");
    printMatrix(*ell_data, limitNum, num_rows);
    printf("new col index for ELL\n");
    printMatrix(*ell_col_index, limitNum, num_rows);
    printf("new data for COO\n");
    printVector(*coo_data, coo_data_cnt);
    printf("new row index for COO\n");
    printVector(*coo_row_index, coo_data_cnt);
    printf("new col index for COO\n");
    printVector(*coo_col_index, coo_data_cnt);
#endif

    free(data_temp);
    free(col_index_temp);

    return coo_data_cnt;
}

void SpMV_COO_seq(float* data, int* col_index, int* row_index, int num_elem, float* x, float* y)
{
    for (int i = 0; i < num_elem; i++)
        y[row_index[i]] += data[i] * x[col_index[i]];
}

__global__
void SpMV_COO(float* data, int* col_index, int* row_index, int num_elem, float* x, float* y)
{
    int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i < num_elem) {
        float dot;
        dot = data[i] * x[col_index[i]];
        atomicAdd(&y[row_index[i]], dot);
    }
}