#include <stdio.h>

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
    }
}

// Instantiate the reduction function for 3 types
template void reduce<int>(int size, int threads, int blocks, int smemSize, int whichKernel,
                          int *d_in, int *d_out);

template void reduce<float>(int size, int threads, int blocks, int smemSize, int whichKernel,
                            float *d_in, float *d_out);

template void reduce<double>(int size, int threads, int blocks, int smemSize, int whichKernel,
                             double *d_in, double *d_out);