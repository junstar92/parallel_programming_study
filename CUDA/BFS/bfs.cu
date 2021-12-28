/*****************************************************************************
 * File:        bfs.cu
 * Description: Implement sequential BFS and parallel BFS
 *              
 * Compile:     nvcc -o bfs bfs.cu -I.. [-DDEBUG]
 * Run:         ./bfs
 * Argument:    n.a
 *****************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <random>
#include <cuda_runtime.h>
#include <queue>

#include <common/common.h>
#include <common/common_string.h>

#define MAX_FRONTIER_SIZE (1<<12)

#define BLOCK_SIZE 512
#define BLOCK_QUEUE_SIZE 32

int initSetting(int* adj);
int initSetting(int* adj, int rows, int cols, int ratio = 10);
void decomposeCSR(int* adj, int rows, int cols, int *data, int *col_index, int *row_ptr);
bool checkValid(int* cpu_output, int* gpu_output, int numVertex);

void BFS_sequential(int source, int* edges, int* dest, int* label);
void BFS(int source, int* adj, int numVertex, int* label);
void BFS_host(int source, int* h_edges, int* h_dest, int* h_label, int numVertex, int numNonzero);

int main(int argc, char** argv)
{
    printf("[BFS...]\n");
#ifdef DEBUG
#undef MAX_FRONTIER_SIZE
#define MAX_FRONTIER_SIZE 9
#endif
    printf("N : %d\n", MAX_FRONTIER_SIZE);
    int* adj;
    adj = (int*)malloc(MAX_FRONTIER_SIZE*MAX_FRONTIER_SIZE*sizeof(int*));
    (void)memset(adj, 0, MAX_FRONTIER_SIZE*MAX_FRONTIER_SIZE*sizeof(int));

    // init setting
#ifdef DEBUG
    int numNonzero = initSetting(adj);
#else
    int numNonzero = initSetting(adj, MAX_FRONTIER_SIZE, MAX_FRONTIER_SIZE, 20);
#endif
#ifdef DEBUG
    for (int i = 0; i < MAX_FRONTIER_SIZE; i++) {
        for (int j = 0; j < MAX_FRONTIER_SIZE; j++)
            printf("%d ", adj[i*MAX_FRONTIER_SIZE + j]);
        printf("\n");
    }
#endif

    // Decompose for CSR
    int* h_data = (int*)malloc(numNonzero*sizeof(int));
    int* h_dest = (int*)malloc(numNonzero*sizeof(int));
    int* h_edges = (int*)malloc((MAX_FRONTIER_SIZE+1)*sizeof(int));
    decomposeCSR(adj, MAX_FRONTIER_SIZE, MAX_FRONTIER_SIZE, h_data, h_dest, h_edges);
#ifdef DEBUG
    // check
    printf("[CSR Format]\n");
    printf("data: ");
    for (int i = 0; i < numNonzero; i++)
        printf("%d ", h_data[i]);
    printf("\n");
    printf("dest: ");
    for (int i = 0; i < numNonzero; i++)
        printf("%d ", h_dest[i]);
    printf("\n");
    printf("edges: ");
    for (int i = 0; i < MAX_FRONTIER_SIZE+1; i++)
        printf("%d ", h_edges[i]);
    printf("\n");
#endif

    int source = 0;
    printf("Source Vertex: %d\n", source);
    int *cpu_output = (int*)malloc(MAX_FRONTIER_SIZE*sizeof(int));
    int *h_label = (int*)malloc(MAX_FRONTIER_SIZE*sizeof(int));

    for (int i = 0; i < MAX_FRONTIER_SIZE; i++)
        cpu_output[i] = -1;
    BFS_sequential(0, h_edges, h_dest, cpu_output);

    // for (int i = 0; i < MAX_FRONTIER_SIZE; i++)
    //     cpu_output[i] = -1;
    // GET_TIME(start);
    // BFS(source, adj, MAX_FRONTIER_SIZE, cpu_output);
    // GET_TIME(finish);
    // printf("BFS with STL Queue: %.6f msec\n", (finish-start)*1000);
#ifdef DEBUG
    printf("Result of sequential BFS\n");
    for (int i = 0; i < MAX_FRONTIER_SIZE; i++)
        printf("%d ", cpu_output[i]);
    printf("\n\n");
#endif

    for (int i = 0; i < MAX_FRONTIER_SIZE; i++)
        h_label[i] = -1;
    BFS_host(source, h_edges, h_dest, h_label, MAX_FRONTIER_SIZE, numNonzero);
#ifdef DEBUG
    printf("Result of parallel BFS\n");
    for (int i = 0; i < MAX_FRONTIER_SIZE; i++)
        printf("%d ", h_label[i]);
    printf("\n\n");
#endif

    printf(checkValid(cpu_output, h_label, MAX_FRONTIER_SIZE) ? "PASSED\n" : "FAILED!\n");

    free(adj);
    free(h_data);
    free(h_dest);
    free(h_edges);
    free(h_label);
    free(cpu_output);

    return 0;
}

int initSetting(int* adj)
{
    adj[0*MAX_FRONTIER_SIZE + 1] = adj[0*MAX_FRONTIER_SIZE + 2] = 1;
    adj[1*MAX_FRONTIER_SIZE + 3] = adj[1*MAX_FRONTIER_SIZE + 4] = 1;
    adj[2*MAX_FRONTIER_SIZE + 5] = adj[2*MAX_FRONTIER_SIZE + 6] = adj[2*MAX_FRONTIER_SIZE + 7] = 1;
    adj[3*MAX_FRONTIER_SIZE + 4] = adj[3*MAX_FRONTIER_SIZE + 8] = 1;
    adj[4*MAX_FRONTIER_SIZE + 5] = adj[4*MAX_FRONTIER_SIZE + 8] = 1;
    adj[5*MAX_FRONTIER_SIZE + 6] = 1;
    adj[6*MAX_FRONTIER_SIZE + 8] = 1;
    adj[7*MAX_FRONTIER_SIZE + 0] = adj[7*MAX_FRONTIER_SIZE + 6] = 1;

    return 15;
}

int initSetting(int* adj, int rows, int cols, int ratio)
{
    std::default_random_engine generator;
    std::uniform_int_distribution<int> dist(0, 100);
    int num_nonzero = 0;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (dist(generator) < ratio) {
                adj[i*cols + j] = 1;
                num_nonzero++;
            }
            else {
                adj[i*cols + j] = 0;
            }
        }
    }

    return num_nonzero;
}

void decomposeCSR(int* adj, int rows, int cols, int *data, int *col_index, int *row_ptr)
{
    int idx = 0;
    for (int i = 0; i < rows; i++) {
        row_ptr[i] = idx;
        for (int j = 0; j < cols; j++) {
            if (adj[i*cols + j] != 0) {
                data[idx] = adj[i*cols + j];
                col_index[idx] = j;
                idx++;
            }
        }
    }
    row_ptr[rows] = idx;
}

bool checkValid(int* cpu_output, int* gpu_output, int numVertex)
{
    for (int i = 0; i < numVertex; i++) {
        if (cpu_output[i] != gpu_output[i])
            return false;
    }

    return true;
}

void insert_frontier(int source, int* frontier, int* frontier_tail)
{
    frontier[*frontier_tail] = source;
    (*frontier_tail)++;
}

void BFS_sequential(int source, int* edges, int* dest, int* label)
{
    int frontier[2][MAX_FRONTIER_SIZE];
    int *c_frontier = frontier[0];
    int c_frontier_tail = 0;
    int *p_frontier = frontier[1];
    int p_frontier_tail = 0;

    insert_frontier(source, p_frontier, &p_frontier_tail);
    label[source] = 0;

    while (p_frontier_tail > 0) {
        for (int f = 0; f < p_frontier_tail; f++) {
            int c_vertex = p_frontier[f];
            for (int i = edges[c_vertex]; i < edges[c_vertex+1]; i++) {
                if (label[dest[i]] == -1) {
                    insert_frontier(dest[i], c_frontier, &c_frontier_tail);
                    label[dest[i]] = label[c_vertex] + 1;
                }
            }
        }

        int *tmp = c_frontier;
        c_frontier = p_frontier;
        p_frontier = tmp;

        p_frontier_tail = c_frontier_tail;
        c_frontier_tail = 0;
    }
}

void BFS(int source, int* adj, int numVertex, int* label)
{
    std::queue<int> q;
    int* visited = (int*)malloc(numVertex*sizeof(int));
    (void)memset(visited, 0, numVertex*sizeof(int));

    q.push(source);
    visited[source] = 1;
    label[source] = 0;
    while (!q.empty()) {
        int qSize = q.size();

        for (int i = 0; i < qSize; i++) {
            int curr = q.front();
            q.pop();

            for (int j = 0; j < numVertex; j++) {
                if (adj[curr*numVertex + j] == 1 && visited[j] == 0) {
                    visited[j] = 1;
                    label[j] = label[curr] + 1;
                    q.push(j);
                }
            }
        }
    }


    free(visited);
}

__global__
void BFS_Bqueue_kernel(int* p_frontier, int* p_frontier_tail, int* c_frontier, int* c_frontier_tail, int* edges, int* dest, int* label, int* visited)
{
    __shared__ int c_frontier_s[BLOCK_QUEUE_SIZE];
    __shared__ int c_frontier_tail_s, our_c_frontier_tail;

    if (threadIdx.x == 0)
        c_frontier_tail_s = 0;
    __syncthreads();

    const int tid = blockIdx.x*blockDim.x + threadIdx.x;
    if (tid < *p_frontier_tail) {
        const int my_vertex = p_frontier[tid];
        for (int i = edges[my_vertex]; i < edges[my_vertex+1]; i++) {
            const int was_visited = atomicExch(&(visited[dest[i]]), 1);
            if (!was_visited) {
                label[dest[i]] = label[my_vertex] + 1;
                const int my_tail = atomicAdd(&c_frontier_tail_s, 1);
                if (my_tail < BLOCK_QUEUE_SIZE) {
                    c_frontier_s[my_tail] = dest[i];
                }
                else {
                    c_frontier_tail_s = BLOCK_QUEUE_SIZE;
                    const int my_global_tail = atomicAdd(c_frontier_tail, 1);
                    c_frontier[my_global_tail] = dest[i];
                }
            }
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        our_c_frontier_tail = atomicAdd(c_frontier_tail, c_frontier_tail_s);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < c_frontier_tail_s; i++) {
        c_frontier[our_c_frontier_tail + i] = c_frontier_s[i];
    }
}

void BFS_host(int source, int* h_edges, int* h_dest, int* h_label, int numVertex, int numNonzero)
{
    // host memory
    int *h_p_frontier = (int*)malloc(numVertex*sizeof(int));
    int *h_c_frontier = (int*)malloc(numVertex*sizeof(int));
    int h_p_frontier_tail = 1;
    int h_c_frontier_tail = 0;
    int *h_visited = (int*)malloc(numVertex*sizeof(int));
    for (int i = 0; i < numVertex; i++)
        h_visited[i] = 0;
    
    // init
    h_label[source] = 0;
    h_visited[source] = 1;
    h_p_frontier[0] = source;

    // allocate device memory
    int *d_edges, *d_dest, *d_label, *d_visited;
    CUDA_CHECK(cudaMalloc((void**)&d_edges, (numVertex+1)*sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_dest, numNonzero*sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_label, numVertex*sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_visited, numVertex*sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_edges, h_edges, (numVertex+1)*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dest, h_dest, numNonzero*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_label, h_label, numVertex*sizeof(int), cudaMemcpyHostToDevice));

    // allocate d_frontier, d_c_frontier_tail, d_p_frontier_tail
    int *d_frontier, *d_c_frontier_tail, *d_p_frontier_tail;
    CUDA_CHECK(cudaMalloc((void**)&d_frontier, 2*numNonzero*sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_c_frontier_tail, sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_p_frontier_tail, sizeof(int)));

    int *d_c_frontier = &d_frontier[0];
    int *d_p_frontier = &d_frontier[numVertex];

    // init
    CUDA_CHECK(cudaMemcpy(d_visited, h_visited, numVertex*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_frontier+numVertex, h_p_frontier, numVertex*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_p_frontier_tail, &h_p_frontier_tail, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_label, h_label, numVertex*sizeof(int), cudaMemcpyHostToDevice));

    while (h_p_frontier_tail > 0) {
        int num_blocks = (h_p_frontier_tail+BLOCK_SIZE-1) / BLOCK_SIZE;

        BFS_Bqueue_kernel<<<num_blocks, BLOCK_SIZE>>>(d_p_frontier, d_p_frontier_tail, d_c_frontier, d_c_frontier_tail, d_edges, d_dest, d_label, d_visited);
        CUDA_CHECK(cudaMemcpy(&h_p_frontier_tail, d_c_frontier_tail, sizeof(int), cudaMemcpyDeviceToHost));

        int* temp = d_c_frontier;
        d_c_frontier = d_p_frontier;
        d_p_frontier = temp;

        CUDA_CHECK(cudaMemcpy(d_p_frontier_tail, d_c_frontier_tail, sizeof(int), cudaMemcpyDeviceToDevice));
        h_c_frontier_tail = 0;
        CUDA_CHECK(cudaMemcpy(d_c_frontier_tail, &h_c_frontier_tail, sizeof(int), cudaMemcpyHostToDevice));
    }

    CUDA_CHECK(cudaMemcpy(h_label, d_label, numVertex*sizeof(int), cudaMemcpyDeviceToHost));

    // free memory
    free(h_p_frontier);
    free(h_c_frontier);
    free(h_visited);
    CUDA_CHECK(cudaFree(d_edges));
    CUDA_CHECK(cudaFree(d_dest));
    CUDA_CHECK(cudaFree(d_label));
    CUDA_CHECK(cudaFree(d_visited));
    CUDA_CHECK(cudaFree(d_frontier));
    CUDA_CHECK(cudaFree(d_c_frontier_tail));
    CUDA_CHECK(cudaFree(d_p_frontier_tail));
}