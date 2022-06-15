#include <iostream>

__host__ __device__ __forceinline__
uint32_t hash(uint32_t x)
{
    x ^= x >> 16;
    x *= 0x85ebca6b;
    x ^= x >> 13;
    x *= 0xc2b2ae35;
    x ^= x >> 16;

    return x;
}

__host__ __device__ __forceinline__
uint64_t permute64(uint64_t x, uint64_t num_iters)
{
    constexpr uint64_t mask = (1UL << 32) - 1;

    for (uint64_t iter = 0; iter < num_iters; iter++) {
        const uint64_t upper = x >> 32;
        const uint64_t lower = x & mask;
        const uint64_t mixer = hash(upper);

        x = upper + ((lower ^ mixer & mask) << 32);
    }

    return x;
}

__host__ __device__ __forceinline__
uint64_t unpermute64(uint64_t x, uint64_t num_iters)
{
    constexpr uint64_t mask = (1UL << 32) - 1;

    for (uint64_t iter = 0; iter < num_iters; iter++) {
        const uint64_t upper = x & mask;
        const uint64_t lower = x >> 32;
        const uint64_t mixer = hash(upper);

        x = (upper << 32) + (lower ^ mixer & mask);
    }

    return x;
}

void encrypt_cpu(uint64_t* data, uint64_t num_entries, uint64_t num_iters, bool parallel=true)
{
    // Use OpenMP to use all available CPU cores.
    #pragma omp parallel for if (parallel)
    for (uint64_t entry = 0; entry < num_entries; entry++) {
        // Permute each data entry the number of iterations and then write result to data.
        data[entry] = permute64(entry, num_iters);
    }
}

__global__
void decrypt_gpu(uint64_t* data, uint64_t num_entries, uint64_t num_iters)
{
    const uint64_t threadID = blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t stride = blockDim.x * gridDim.x;

    for (uint64_t entry = threadID; entry < num_entries; entry += stride) {
        // Unpermute each data entry the number of iterations then write result to data.
        data[entry] = unpermute64(data[entry], num_iters);
    }
}

bool check_result_cpu(uint64_t* data, uint64_t num_entries, bool parallel=true)
{
    uint64_t counter = 0;

    #pragma omp parallel for reduction(+: counter) if (parallel)
    for (uint64_t entry = 0; entry < num_entries; entry++) {
        counter += data[entry] == entry;
    }

    return counter == num_entries;
}

// Timer will be used throughout to benchmarks sections of the application,
// as well as total time on the GPU(s).
class Timer {

    float time;
    const uint64_t gpu;
    cudaEvent_t ying, yang;

public:

    Timer (uint64_t gpu=0) : gpu(gpu) {
        cudaSetDevice(gpu);
        cudaEventCreate(&ying);
        cudaEventCreate(&yang);
    }

    ~Timer ( ) {
        cudaSetDevice(gpu);
        cudaEventDestroy(ying);
        cudaEventDestroy(yang);
    }

    void start ( ) {
        cudaSetDevice(gpu);
        cudaEventRecord(ying, 0);
    }

    void stop (std::string label) {
        cudaSetDevice(gpu);
        cudaEventRecord(yang, 0);
        cudaEventSynchronize(yang);
        cudaEventElapsedTime(&time, ying, yang);
        std::cout << "TIMING: " << time << " ms (" << label << ")" << std::endl;
    }
};

uint64_t sdiv (uint64_t a, uint64_t b) {
    return (a+b-1)/b;
}

int main()
{
    Timer timer, overall;

    const uint64_t num_entries = 1UL << 26;
    const uint64_t num_iters = 1UL << 10;
    const bool openmp = true;

    // Set number of avaiable GPUs and number of streams.
    const uint64_t num_gpus = 4;
    const uint64_t num_streams = 32;

    // Get chunk size using round up division.
    const uint64_t stream_chunk_size = sdiv(sdiv(num_entries, num_gpus), num_streams);
    // It will be helpful to also to have handy the chunk size for an entire GPU.
    const uint64_t gpu_chunk_size = stream_chunk_size * num_streams;

    // 2D array containing number of streams for each GPU.
    cudaStream_t streams[num_gpus][num_streams];
    for (uint64_t gpu = 0; gpu < num_gpus; gpu++) {
        // set as active device
        cudaSetDevice(gpu);
        for (uint64_t stream = 0; stream < num_streams; stream++) {
            // create and store its number of streams
            cudaStreamCreate(&streams[gpu][stream]);
        }
    }    

    // Store GPU data pointers in an array.
    uint64_t *data_cpu, *data_gpu[num_gpus];
    cudaMallocHost(&data_cpu, sizeof(uint64_t) * num_entries);
    for (uint64_t gpu = 0; gpu < num_gpus; gpu++) {
        // set GPU as active
        cudaSetDevice(gpu);
        // get width of this GPUs data chunk
        const uint64_t lower = gpu_chunk_size * gpu;
        const uint64_t upper = min(lower + gpu_chunk_size, num_entries);
        const uint64_t width = upper - lower;

        // allocate data for this GPU.
        cudaMalloc(&data_gpu[gpu], sizeof(uint64_t) * width);
    }

    // encrypt data
    encrypt_cpu(data_cpu, num_entries, num_iters, openmp);

    overall.start();
    // For each gpu...
    for (uint64_t gpu = 0; gpu < num_gpus; gpu++) {
        cudaSetDevice(gpu);
        // For each stream (on each GPU)...
        for (uint64_t stream = 0; stream < num_streams; stream++) {
            // Calculate index offset for this stream's chunk of data within the GPU's chunk of data...
            const uint64_t stream_offset = stream_chunk_size * stream;
            
            // ...get the lower index within all data, and width of this stream's data chunk...
            const uint64_t lower = gpu_chunk_size * gpu + stream_offset;
            const uint64_t upper = min(lower + stream_chunk_size, num_entries);
            const uint64_t width = upper - lower;

            // ...perform async HtoD memory copy...
            cudaMemcpyAsync(data_gpu[gpu] + stream_offset, // This stream's data within this GPU's data.
                            data_cpu + lower,              // This stream's data within all CPU data.
                            sizeof(uint64_t) * width,      // This stream's chunk size worth of data.
                            cudaMemcpyHostToDevice,
                            streams[gpu][stream]);         // Using this stream for this GPU.

            decrypt_gpu<<<80*32, 64, 0, streams[gpu][stream]>>>    // Using this stream for this GPU.
                (data_gpu[gpu]+stream_offset,                      // This stream's data within this GPU's data.
                 width,                                            // This stream's chunk size worth of data.
                 num_iters);

            cudaMemcpyAsync(data_cpu + lower,              // This stream's data within all CPU data.
                            data_gpu[gpu] + stream_offset, // This stream's data within this GPU's data.
                            sizeof(uint64_t) * width,
                            cudaMemcpyDeviceToHost,
                            streams[gpu][stream]);         // Using this stream for this GPU.
        }
    }

    // Synchronize streams to block on memory transfer before checking on host.
    for (uint64_t gpu = 0; gpu < num_gpus; gpu++) {
        cudaSetDevice(gpu);
        for (uint64_t stream = 0; stream < num_streams; stream++) {
            cudaStreamSynchronize(streams[gpu][stream]);
        }
    }

    // Stop timer for total time on GPU(s).
    overall.stop("total time on GPU");

    // Check results on CPU.
    const bool success = check_result_cpu(data_cpu, num_entries, openmp);
    std::cout << "STATUS: test " << (success ? "passed" : "failed") << std::endl;

    for (uint64_t gpu = 0; gpu < num_gpus; gpu++) {
        cudaSetDevice(gpu);
        for (uint64_t stream = 0; stream < num_streams; stream++) {
            cudaStreamDestroy(streams[gpu][stream]);
        }
    }
    // Free memory
    cudaFreeHost(data_cpu);
    for (uint64_t gpu = 0; gpu < num_gpus; gpu++) {
        cudaSetDevice(gpu);
        cudaFree(data_gpu[gpu]);
    }
}