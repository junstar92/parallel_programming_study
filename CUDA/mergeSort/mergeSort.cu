/*****************************************************************************
 * File:        mergeSort.cu
 * Description: 
 *              
 * Compile:     nvcc -o mergeSort mergeSort.cu -I..
 * Run:         ./mergeSort
 * Argument:
 *      "--n=<N>"           : Specify the number of elements (default: 1<<24)
 *****************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <random>
#include <cuda_runtime.h>

#include <common/common.h>
#include <common/common_string.h>

void seqMergeSort(int* in, int s, int e);
void sequentialMerge(int* A, int nA, int* B, int nB);
int co_rank(int k, int* A, int nA, int* B, int nB);


int main(int argc, char** argv)
{
    printf("[Parallel Merge Sort...]\n\n");
    int n = 100;

    if (checkCmdLineFlag(argc, (const char **)argv, "n")) {
        n = getCmdLineArgumentInt(argc, (const char **)argv, "n");
    }

    printf("Size of input array: %d\n", n);

    int *h_in, *h_out;
    h_in = (int*)malloc(n*sizeof(int));
    h_out = (int*)malloc(n*sizeof(int));

    for (int i = 0; i < n; i++)
        h_in[i] = rand() % 200;
    (void)memcpy(h_out, h_in, n*sizeof(int));

    seqMergeSort(h_out, 0, n);

#ifndef DEBUG
    for (int i = 0; i < n; i++)
        printf("%d ", h_out[i]);
    printf("\n");
#endif

    free(h_in);
    free(h_out);
}

void seqMergeSort(int* in, int s, int e)
{
    if (s+1 < e) {
        int mid = (s + e) / 2;
        seqMergeSort(in, s, mid);
        seqMergeSort(in, mid, e);
        //sequentialMerge(in, s, mid, e);
        sequentialMerge(in+s, mid-s, in+mid, e-mid);
    }
}

void sequentialMerge(int* A, int nA, int* B, int nB)
{
    int *tmpA, *tmpB;
    tmpA = (int*)malloc(nA*sizeof(int));
    tmpB = (int*)malloc(nB*sizeof(int));

    for (int i = 0; i < nA; i++)
        tmpA[i] = A[i];
    for (int i = 0; i < nB; i++)
        tmpB[i] = B[i];
    
    int *C = A;

    int iA = 0, iB = 0, iC = 0;
    while ((iA < nA) && (iB < nB)) {
        if (tmpA[iA] <= tmpB[iB]) {
            C[iC++] = tmpA[iA++];
        }
        else {
            C[iC++] = tmpB[iB++];
        }
    }

    if (iA == nA) {
        while (iB < nB)
            C[iC++] = tmpB[iB++];
    }
    else {
        while (iA < nA)
            C[iC++] = tmpA[iA++];
    }

    free(tmpA);
    free(tmpB);
}

int co_rank(int k, int* A, int nA, int* B, int nB)
{
    int i = (k < nA) ? k : nA; // i = min(k, nA);
    int j = k-i;
    int i_low = (0 > (k-nB)) ? 0 : k-nB; // i_low = max(0, k-nB);
    int j_low = (0 > (k-nA)) ? 0 : k-nA; // j_low = max(0, k-nA);
    int delta;
    bool active = true;

    while (active) {
        if (i > 0 && j < nB && A[i-1] > B[j]) {
            delta = ((i - i_low + 1) >> 1);
            j_low = j;
            j = j + delta;
            i = i - delta;
        }
        else if (j > 0 && i < nA && B[j-1] >= A[i]) {
            delta = ((j - j_low + 1) >> 1);
            i_low = i;
            i = i + delta;
            j = j - delta;
        }
        else {
            active = false;
        }
    }

    return i;
}

__global__
void merge_basic_kernel(int* A, int nA, int* B, int nB)
{
    int tid = blockDim.x*blockIdx.x + threadIdx.x;
    int k_curr = tid * ceil((nA+nB)/(blockDim.x*gridDim.x));
    int k_next = min((tid+1) * ceil((nA+nB)/(blockDim.x*gridDim.x)), nA+nB);
    int i_curr = co_rank(k_curr, A, nA, B, nB);
    int i_next = co_rank(k_next, A, nA, B, nB);
    int j_curr = k_curr - i_curr;
    int j_next = k_next - i_next;

    sequentialMerge(&A[i_curr], i_next-i_curr, &B[j_curr], j_next-j_curr);
}