/*****************************************************************************
 * File:        transposeRectangular.cu
 * Description: This is an example of kernels for transposing a rectangular
 *              host array using a variety of optimizations, including shared
 *              memory, unrolling, and memory padding.
 *              
 * Compile:     nvcc -o transposeRectangular transposeRectangular.cu -I..
 * Run:         ./transposeRectangular
 *****************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "common/common.h"

#define BDIMX 32
#define BDIMY 16
#define INDEX(ROW, COL, INNER) ((ROW) * (INNER) + (COL))
#define IPAD 2

void initialData(float* in, const int size)
{
    for (int i = 0; i < size; i++)
        in[i] = (rand() & 0xFF) / 10.f;
}

void printData(float* in, const int size)
{
    for (int i = 0; i < size; i++)
        printf("%3.0f ", in[i]);
    printf("\n");
}

void checkResult(float* hostRef, float* gpuRef, int rows, int cols)
{
    double epsilon = 1.0e-8;
    bool match = true;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            int index = INDEX(i, j, cols);
            if (abs(hostRef[index] - gpuRef[index]) > epsilon) {
                printf("different on (%d, %d) (offset=%d) element in transposed matrix: host %f gpu %f\n",
                            i, j, index, hostRef[index], gpuRef[index]);
                match = false;
                break;
            }
        }
        if (!match)
            break;
    }
    if (!match)
        printf("Arrays do not match.\n\n");
}

void transposeHost(float* in, float* out, const int nRows, const int nCols)
{
    for (int iy = 0; iy < nRows; iy++) {
        for (int ix = 0; ix < nCols; ix++) {
            out[INDEX(ix, iy, nRows)] = in[INDEX(iy, ix, nCols)];
        }
    }
}

__global__
void copyGmem(float* in, float* out, const int nRows, const int nCols)
{
    // matrix coordinate (ix, iy)
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    // transpose with boundary test
    if (row < nRows && col < nCols) {
        out[row*nCols + col] = in[row*nCols + col];
    }
}

__global__
void naiveGmem(float* in, float* out, const int nRows, const int nCols)
{
    // matrix coordinate (ix, iy)
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    // transpose with boundary test
    if (row < nRows && col < nCols) {
        out[col*nRows + row] = in[row*nCols + col];
    }
}

__global__
void transposeSmem(float* in, float* out, const int nRows, const int nCols)
{
    // static shared memory
    __shared__ float tile[BDIMY][BDIMX];

    // coordinate in original matrix
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    // linear global memory index for original matrix
    unsigned int offset = row * nCols + col;

    if (row < nRows && col < nCols) {
        // load data from global memory to shared memory
        tile[threadIdx.y][threadIdx.x] = in[offset];
    }
    __syncthreads();

    // thread index in transposed block
    unsigned int bidx, irow, icol;
    bidx = threadIdx.y * blockDim.x + threadIdx.x;
    irow = bidx / blockDim.y;
    icol = bidx % blockDim.y;

    // coordinate in transposed matrix
    row = blockIdx.x * blockDim.x + irow;
    col = blockIdx.y * blockDim.y + icol;

    // linear global memory index for transposed matrix
    unsigned int transposed_offset = row * nRows + col;
    __syncthreads();

    // transpose with boundary test
    if (row < nCols && col < nRows) {
        // store data to global memory from shared memory
        out[transposed_offset] = tile[icol][irow];
    }
}

__global__
void transposeSmemPad(float* in, float* out, const int nRows, const int nCols)
{
    // static shared memory with padding
    __shared__ float tile[BDIMY][BDIMX + IPAD];

    // coordinate in original matrix
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    // linear global memory index for original matrix
    unsigned int offset = row * nCols + col;

    if (row < nRows && col < nCols) {
        // load data from global memory to shared memory
        tile[threadIdx.y][threadIdx.x] = in[offset];
    }
    __syncthreads();

    // thread index in transposed block
    unsigned int bidx, irow, icol;
    bidx = threadIdx.y * blockDim.x + threadIdx.x;
    irow = bidx / blockDim.y;
    icol = bidx % blockDim.y;

    // coordinate in transposed matrix
    row = blockIdx.x * blockDim.x + irow;
    col = blockIdx.y * blockDim.y + icol;

    // linear global memory index for transposed matrix
    unsigned int transposed_offset = row * nRows + col;
    __syncthreads();

    // transpose with boundary test
    if (row < nCols && col < nRows) {
        // store data to global memory from shared memory
        out[transposed_offset] = tile[icol][irow];
    }
}

__global__
void transposeSmemUnrollPad(float* in, float* out, const int nRows, const int nCols)
{
    // static shared memory with padding
    __shared__ float tile[BDIMY * (2 * BDIMX + IPAD)];

    // coordinate in original matrix
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = 2 * blockIdx.x * blockDim.x + threadIdx.x;

    // linear global memory index for original matrix
    unsigned int offset = row * nCols + col;


    // thread index in transposed block
    unsigned int bidx = threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int irow = bidx / blockDim.y;
    unsigned int icol = bidx % blockDim.y;
    
    // coordinate in transposed matrix
    unsigned int transposed_row = 2 * blockIdx.x * blockDim.x + irow;
    unsigned int transposed_col = blockIdx.y * blockDim.y + icol;

    // linear global memory index for transposed matrix
    unsigned int transposed_offset = transposed_row * nRows + transposed_col;

    if (col + BDIMX < nCols && row < nRows) {
        // load two rows from global memory to shared memory
        unsigned int row_idx = threadIdx.y * (2 * BDIMX + IPAD) + threadIdx.x;
        tile[row_idx] = in[offset];
        tile[row_idx + BDIMX] = in[offset + BDIMX];
    }

    __syncthreads();

    if (transposed_row + BDIMX < nCols && transposed_col < nRows) {
        // store two rows to global memory from two columns of shared memory
        unsigned int col_idx = icol * (2 * BDIMX + IPAD) + irow;
        out[transposed_offset] = tile[col_idx];
        out[transposed_offset + nRows * BDIMX] = tile[col_idx + BDIMX];
    }
}

__global__
void transposeSmemUnrollPadDyn(float* in, float* out, const int nRows, const int nCols)
{
    // static shared memory with padding
    extern __shared__ float tile[];

    // coordinate in original matrix
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = 2 * blockIdx.x * blockDim.x + threadIdx.x;

    // linear global memory index for original matrix
    unsigned int offset = row * nCols + col;


    // thread index in transposed block
    unsigned int bidx = threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int irow = bidx / blockDim.y;
    unsigned int icol = bidx % blockDim.y;
    
    // coordinate in transposed matrix
    unsigned int transposed_row = 2 * blockIdx.x * blockDim.x + irow;
    unsigned int transposed_col = blockIdx.y * blockDim.y + icol;

    // linear global memory index for transposed matrix
    unsigned int transposed_offset = transposed_row * nRows + transposed_col;

    if (col + BDIMX < nCols && row < nRows) {
        // load two rows from global memory to shared memory
        unsigned int row_idx = threadIdx.y * (2 * BDIMX + IPAD) + threadIdx.x;
        tile[row_idx] = in[offset];
        tile[row_idx + BDIMX] = in[offset + BDIMX];
    }

    __syncthreads();

    if (transposed_row + BDIMX < nCols && transposed_col < nRows) {
        // store two rows to global memory from two columns of shared memory
        unsigned int col_idx = icol * (2 * BDIMX + IPAD) + irow;
        out[transposed_offset] = tile[col_idx];
        out[transposed_offset + nRows * BDIMX] = tile[col_idx + BDIMX];
    }
}

int main(int argc, char** argv)
{
    // setup device
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("Starting transpose at device %d: %s\n", dev, deviceProp.name);
    cudaSetDevice(dev);

    // setup array size 4096
    int nRows = 1 << 12;
    int nCols = 1 << 12;

    if (argc > 1)
        nRows = atoi(argv[1]);
    if (argc > 2)
        nCols = atoi(argv[2]);
    
    printf("  with matrix nRows %d nCols %d\n", nRows, nCols);
    size_t nCells = nRows * nCols;
    size_t nBytes = nCells * sizeof(float);

    // execution configuration
    dim3 block(BDIMX, BDIMY);
    dim3 grid((nCols + block.x - 1) / block.x, (nRows + block.y - 1) / block.y);
    dim3 grid2((grid.x + 2 - 1) / 2, grid.y);

    // allocate host memory
    float *h_A = (float*)malloc(nBytes);
    float *hostRef = (float*)malloc(nBytes);
    float *gpuRef = (float*)malloc(nBytes);

    // initialize host array
    initialData(h_A, nRows * nCols);

    // transpose at host side
    transposeHost(h_A, hostRef, nRows, nCols);

    // allocate device memory
    float *d_A, *d_C;
    CUDA_CHECK(cudaMalloc((void**)&d_A, nBytes));
    CUDA_CHECK(cudaMalloc((void**)&d_C, nBytes));

    // copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));

    double start, finish, elaps;
    float bandwidth;
    
    // copy gmem
    CUDA_CHECK(cudaMemset(d_C, 0, nBytes));
    memset(gpuRef, 0, nBytes);

    GET_TIME(start);
    copyGmem<<<grid, block>>>(d_A, d_C, nRows, nCols);
    CUDA_CHECK(cudaDeviceSynchronize());
    GET_TIME(finish);
    elaps = finish-start;

    CUDA_CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
    bandwidth = 2 * nCells * sizeof(float) / 1.0e9 / elaps;
    printf("copyGmem elapsed %f sec <<< grid (%d,%d) block (%d,%d) >>> effective bandwidth %f GB/s\n",
            elaps, grid.x, grid.y, block.x, block.y, bandwidth);

    // transpose gmem
    CUDA_CHECK(cudaMemset(d_C, 0, nBytes));
    memset(gpuRef, 0, nBytes);

    GET_TIME(start);
    naiveGmem<<<grid, block>>>(d_A, d_C, nRows, nCols);
    CUDA_CHECK(cudaDeviceSynchronize());
    GET_TIME(finish);
    elaps = finish-start;

    CUDA_CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
    bandwidth = 2 * nCells * sizeof(float) / 1.0e9 / elaps;
    printf("naiveGmem elapsed %f sec <<< grid (%d,%d) block (%d,%d) >>> effective bandwidth %f GB/s\n",
            elaps, grid.x, grid.y, block.x, block.y, bandwidth);
    checkResult(hostRef, gpuRef, nRows, nCols);

    // transpose smem
    CUDA_CHECK(cudaMemset(d_C, 0, nBytes));
    memset(gpuRef, 0, nBytes);

    GET_TIME(start);
    transposeSmem<<<grid, block>>>(d_A, d_C, nRows, nCols);
    CUDA_CHECK(cudaDeviceSynchronize());
    GET_TIME(finish);
    elaps = finish-start;

    CUDA_CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
    bandwidth = 2 * nCells * sizeof(float) / 1.0e9 / elaps;
    printf("transposeSmem elapsed %f sec <<< grid (%d,%d) block (%d,%d) >>> effective bandwidth %f GB/s\n",
            elaps, grid.x, grid.y, block.x, block.y, bandwidth);
    checkResult(hostRef, gpuRef, nRows, nCols);

    // transpose smem with padding
    CUDA_CHECK(cudaMemset(d_C, 0, nBytes));
    memset(gpuRef, 0, nBytes);

    GET_TIME(start);
    transposeSmemPad<<<grid, block>>>(d_A, d_C, nRows, nCols);
    CUDA_CHECK(cudaDeviceSynchronize());
    GET_TIME(finish);
    elaps = finish-start;

    CUDA_CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
    bandwidth = 2 * nCells * sizeof(float) / 1.0e9 / elaps;
    printf("transposeSmemPad elapsed %f sec <<< grid (%d,%d) block (%d,%d) >>> effective bandwidth %f GB/s\n",
            elaps, grid.x, grid.y, block.x, block.y, bandwidth);
    checkResult(hostRef, gpuRef, nRows, nCols);

    // transpose smem unrolling with padding
    CUDA_CHECK(cudaMemset(d_C, 0, nBytes));
    memset(gpuRef, 0, nBytes);

    GET_TIME(start);
    transposeSmemUnrollPad<<<grid2, block>>>(d_A, d_C, nRows, nCols);
    CUDA_CHECK(cudaDeviceSynchronize());
    GET_TIME(finish);
    elaps = finish-start;

    CUDA_CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
    bandwidth = 2 * nCells * sizeof(float) / 1.0e9 / elaps;
    printf("transposeSmemUnrollPad elapsed %f sec <<< grid (%d,%d) block (%d,%d) >>> effective bandwidth %f GB/s\n",
            elaps, grid2.x, grid2.y, block.x, block.y, bandwidth);
    checkResult(hostRef, gpuRef, nRows, nCols);

    // transpose smem unrolling with padding
    CUDA_CHECK(cudaMemset(d_C, 0, nBytes));
    memset(gpuRef, 0, nBytes);

    GET_TIME(start);
    transposeSmemUnrollPadDyn<<<grid2, block, (BDIMX * 2 + IPAD) * BDIMY * sizeof(float)>>>(d_A, d_C, nRows, nCols);
    CUDA_CHECK(cudaDeviceSynchronize());
    GET_TIME(finish);
    elaps = finish-start;

    CUDA_CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
    bandwidth = 2 * nCells * sizeof(float) / 1.0e9 / elaps;
    printf("transposeSmemUnrollPadDyn elapsed %f sec <<< grid (%d,%d) block (%d,%d) >>> effective bandwidth %f GB/s\n",
            elaps, grid2.x, grid2.y, block.x, block.y, bandwidth);
    checkResult(hostRef, gpuRef, nRows, nCols);

    // free host and device memory
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A);
    free(hostRef);
    free(gpuRef);

    // reset device
    CUDA_CHECK(cudaDeviceReset());
    return 0;
}