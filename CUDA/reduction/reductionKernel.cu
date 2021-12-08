#include <stdio.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// Wrapper function for kernel launch
template <class T>
void reduce(int size, int threads, int blocks, int smemSize, int whichKernel, T *d_in, T *d_out);

// Sum Reduction Kernel functions
template<class T>
__global__ void sumReduce1(T* g_in, T* g_out, unsigned int size);
template<class T>
__global__ void sumReduce2(T* g_in, T* g_out, unsigned int size);
template<class T>
__global__ void sumReduce3(T* g_in, T* g_out, unsigned int size);
template<class T>
__global__ void sumReduce4(T* g_in, T* g_out, unsigned int size);
template<class T, unsigned int blockSize>
__global__ void sumReduce5(T* g_in, T* g_out, unsigned int size);


// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
template <class T>
struct SharedMemory
{
    __device__ inline operator T *()
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }

    __device__ inline operator const T *() const
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }
};

// specialize for double to avoid unaligned memory
// access compile errors
template <>
struct SharedMemory<double>
{
    __device__ inline operator double *()
    {
        extern __shared__ double __smem_d[];
        return (double *)__smem_d;
    }

    __device__ inline operator const double *() const
    {
        extern __shared__ double __smem_d[];
        return (double *)__smem_d;
    }
};

template<class T>
__global__ void sumReduce1(T* g_in, T* g_out, unsigned int size)
{
    T *sdata = SharedMemory<T>();

    unsigned int t = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    sdata[t] = (i < size) ? g_in[i] : 0;
    __syncthreads();

    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        if (t % (2*stride) == 0)
            sdata[t] += sdata[t+stride];
        __syncthreads();
    }

    if (t == 0)
        g_out[blockIdx.x] = sdata[0];
}

template<class T>
__global__ void sumReduce2(T* g_in, T* g_out, unsigned int size)
{
    T *sdata = SharedMemory<T>();

    unsigned int t = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    sdata[t] = (i < size) ? g_in[i] : 0;
    __syncthreads();

    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        int index = 2 * stride * t;

        if (index < blockDim.x)
            sdata[index] += sdata[index+stride];
        __syncthreads();
    }

    if (t == 0)
        g_out[blockIdx.x] = sdata[0];
}

template<class T>
__global__ void sumReduce3(T* g_in, T* g_out, unsigned int size)
{
    T *sdata = SharedMemory<T>();

    unsigned int t = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    sdata[t] = (i < size) ? g_in[i] : 0;
    __syncthreads();

    for (unsigned int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (t < stride)
            sdata[t] += sdata[t+stride];
        __syncthreads();
    }

    if (t == 0)
        g_out[blockIdx.x] = sdata[0];
}

template<class T>
__global__ void sumReduce4(T* g_in, T* g_out, unsigned int size)
{
    T *sdata = SharedMemory<T>();

    unsigned int t = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x*2) + threadIdx.x;

    T mySum = (i < size) ? g_in[i] : 0;
    if (i + blockDim.x < size)
        mySum += g_in[i + blockDim.x];
    sdata[t] = mySum;
    __syncthreads();

    for (unsigned int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (t < stride)
            sdata[t] = mySum = mySum + sdata[t+stride];
        __syncthreads();
    }

    if (t == 0)
        g_out[blockIdx.x] = mySum;
}

template<class T, unsigned int blockSize>
__global__ void sumReduce5(T* g_in, T* g_out, unsigned int size)
{
    cg::thread_block cta = cg::this_thread_block();
    T *sdata = SharedMemory<T>();

    unsigned int t = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    T mySum = (i < size) ? g_in[i] : 0;
    if (i + blockSize < size)
        mySum += g_in[i + blockSize];

    sdata[t] = mySum;
    cg::sync(cta);

    for (unsigned int stride = blockDim.x/2; stride > 32; stride >>= 1) {
        if (t < stride)
            sdata[t] = mySum = mySum + sdata[t+stride];
        cg::sync(cta);
    }

    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);
    if (cta.thread_rank() < 32) {
        // Fetch final intermediate sum from 2nd warp
        if (blockSize >= 64)
            mySum += sdata[t + 32];
        // Reduce final warp using shuffle
        for (int offset = tile32.size() / 2; offset > 0; offset /= 2) {
            mySum += tile32.shfl_down(mySum, offset);
         }
    }

    // write result for this block to global memory
    if (cta.thread_rank() == 0)
        g_out[blockIdx.x] = mySum;
}

template <class T>
void reduce(int size, int threads, int blocks, int smemSize, int whichKernel, T *d_in, T *d_out)
{
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);

    switch (whichKernel) {
        case 1:
            sumReduce1<T><<<dimGrid, dimBlock, smemSize>>>(d_in, d_out, size);
            break;
        case 2:
            sumReduce2<T><<<dimGrid, dimBlock, smemSize>>>(d_in, d_out, size);
            break;
        case 3:
            sumReduce3<T><<<dimGrid, dimBlock, smemSize>>>(d_in, d_out, size);
            break;
        case 4:
            sumReduce4<T><<<dimGrid, dimBlock, smemSize>>>(d_in, d_out, size);
            break;
        case 5:
            switch (threads) {
                case 512:
                    sumReduce5<T, 512><<<dimGrid, dimBlock, smemSize>>>(d_in, d_out, size);
                    break;
                case 256:
                    sumReduce5<T, 256><<<dimGrid, dimBlock, smemSize>>>(d_in, d_out, size);
                    break;
                case 128:
                    sumReduce5<T, 128><<<dimGrid, dimBlock, smemSize>>>(d_in, d_out, size);
                    break;
                case 64:
                    sumReduce5<T, 64><<<dimGrid, dimBlock, smemSize>>>(d_in, d_out, size);
                    break;
                case 32:
                    sumReduce5<T, 32><<<dimGrid, dimBlock, smemSize>>>(d_in, d_out, size);
                    break;
                case 16:
                    sumReduce5<T, 16><<<dimGrid, dimBlock, smemSize>>>(d_in, d_out, size);
                    break;
                case 8:
                    sumReduce5<T, 8><<<dimGrid, dimBlock, smemSize>>>(d_in, d_out, size);
                    break;
                case 4:
                    sumReduce5<T, 4><<<dimGrid, dimBlock, smemSize>>>(d_in, d_out, size);
                    break;
                case 2:
                    sumReduce5<T, 2><<<dimGrid, dimBlock, smemSize>>>(d_in, d_out, size);
                    break;
                case 1:
                    sumReduce5<T, 1><<<dimGrid, dimBlock, smemSize>>>(d_in, d_out, size);
                    break;
            }
            break;
    }
}

// Instantiate the reduction function for 3 types
template void reduce<int>(int size, int threads, int blocks, int smemSize, int whichKernel,
                          int *d_in, int *d_out);

template void reduce<float>(int size, int threads, int blocks, int smemSize, int whichKernel,
                            float *d_in, float *d_out);

template void reduce<double>(int size, int threads, int blocks, int smemSize, int whichKernel,
                             double *d_in, double *d_out);