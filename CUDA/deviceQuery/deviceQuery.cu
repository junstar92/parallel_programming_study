/*****************************************************************************
 * File:        deviceQuery.cu
 * Description: Query device information
 *              
 * Compile:     nvcc -o deviceQuery deviceQuery.cu -I.. -lcuda
 * Run:         ./deviceQuery <image file path>
 *****************************************************************************/
#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <common/common.h>

int _ConvertSMVer2Cores(int major, int minor);

int main(int argc, char **argv)
{
    printf("%s Starting...\n\n", argv[0]);
    printf(" CUDA Device Query (Runtime API) version (CUDART static linking)\n\n");

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0)
    {
        printf("There are no available device(s) that support CUDA\n");
    }
    else
    {
        printf("Detected %d CUDA Capable device(s)\n", deviceCount);
    }

    int driverVersion = 0, runtimeVersion = 0;
    for (int dev = 0; dev < deviceCount; ++dev)
    {
        CUDA_CHECK(cudaSetDevice(dev));
        cudaDeviceProp deviceProp;
        CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, dev));

        printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);

        // driver and runtime version
        cudaDriverGetVersion(&driverVersion);
        cudaRuntimeGetVersion(&runtimeVersion);
        printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n",
               driverVersion / 1000, (driverVersion % 100) / 10, runtimeVersion / 1000, (runtimeVersion % 100) / 10);
        printf("  CUDA Capability Major/Minor version number:    %d.%d\n", deviceProp.major, deviceProp.minor);

        // Global Memory
        printf("  Total amount of global memory:                 %.0f MBytes (%llu bytes)\n",
               static_cast<float>(deviceProp.totalGlobalMem / 1048576.f),
               (unsigned long long)deviceProp.totalGlobalMem);

        //
        printf("  (%3d) Multiprocessors, (%3d) CUDA Cores/MP:    %d CUDA Cores\n",
               deviceProp.multiProcessorCount,
               _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
               _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount);

        printf("  GPU Max Clock rate:                            %.0f MHz (%0.2f GHz)\n",
               deviceProp.clockRate * 1e-3f, deviceProp.clockRate * 1e-6f);

#if (CUDART_VERSION >= 5000)
        printf("  Memory Clock rate:                             %.0f MHz\n", deviceProp.memoryClockRate * 1e-3f);
        printf("  Memory Bus Width:                              %d-bit\n", deviceProp.memoryBusWidth);

        if (deviceProp.l2CacheSize)
        {
            printf("  L2 Cache Size:                                 %d bytes (%.0f MBytes)\n",
                deviceProp.l2CacheSize, deviceProp.l2CacheSize / 1048576.f);
        }
#else
        // This only available in CUDA 4.0-4.2
        // (but these were only exposed in the CUDA Driver API)
        int memoryClock;
        cuDeviceGetAttribute(&memoryClock, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, dev);
        printf("  Memory Clock rate:                             %.0f MHz\n", memoryClock * 1e-3f);

        int memBusWidth;
        cuDeviceGetAttribute(&memBusWidth, CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, dev);
        printf("  Memory Bus Width:                              %d-bit\n", memBusWidth);

        int L2CacheSize;
        cuDeviceGetAttribute(&L2CacheSize, CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, dev);
        printf("  L2 Cache Size:                                 %d bytes\n", L2CacheSize);
#endif
        // Maximum Texture Dimension
        printf("  Maximum Texture Dimension Size (x,y,z)         1D=(%d), 2D=(%d, %d), 3D=(%d, %d, %d)\n",
               deviceProp.maxTexture1D, deviceProp.maxTexture2D[0], deviceProp.maxTexture2D[1],
               deviceProp.maxTexture3D[0], deviceProp.maxTexture3D[1], deviceProp.maxTexture3D[2]);

        printf("  Total amount of constant memory:               %zu KBytes (%zu bytes)\n",
               deviceProp.totalConstMem / 1024, deviceProp.totalConstMem);
        printf("  Total amount of shared memory per block:       %zu KBytes (%zu bytes)\n",
               deviceProp.sharedMemPerBlock / 1024, deviceProp.sharedMemPerBlock);
        printf("  Total shared memory per multiprocessor:        %zu KBytes (%zu bytes)\n",
               deviceProp.sharedMemPerMultiprocessor / 1024, deviceProp.sharedMemPerMultiprocessor);
        printf("  Total number of registers available per block: %d\n",
               deviceProp.regsPerBlock);
        printf("  Warp size:                                     %d\n",
               deviceProp.warpSize);
        
        printf("  Maximum number of blocks per multiprocessor:   %d\n",
               deviceProp.maxBlocksPerMultiProcessor);
        printf("  Maximum number of threads per multiprocessor:  %d\n",
               deviceProp.maxThreadsPerMultiProcessor);
        printf("  Maximum number of threads per block:           %d\n",
               deviceProp.maxThreadsPerBlock);
        printf("  Max dimension size of a thread block (x,y,z):  (%d, %d, %d)\n",
               deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1],
               deviceProp.maxThreadsDim[2]);
        printf("  Max dimension size of a grid size    (x,y,z):  (%d, %d, %d)\n",
               deviceProp.maxGridSize[0], deviceProp.maxGridSize[1],
               deviceProp.maxGridSize[2]);
    }
}

int _ConvertSMVer2Cores(int major, int minor)
{
    // Defines for GPU Architecture types (using the SM version to determine
    // the # of cores per SM
    typedef struct
    {
        int SM; // 0xMm (hexidecimal notation), M = SM Major version,
        // and m = SM minor version
        int Cores;
    } sSMtoCores;

    sSMtoCores nGpuArchCoresPerSM[] = {
        {0x30, 192},
        {0x32, 192},
        {0x35, 192},
        {0x37, 192},
        {0x50, 128},
        {0x52, 128},
        {0x53, 128},
        {0x60, 64},
        {0x61, 128},
        {0x62, 128},
        {0x70, 64},
        {0x72, 64},
        {0x75, 64},
        {0x80, 64},
        {0x86, 128},
        {-1, -1}};

    int index = 0;

    while (nGpuArchCoresPerSM[index].SM != -1)
    {
        if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor))
        {
            return nGpuArchCoresPerSM[index].Cores;
        }

        index++;
    }

    // If we don't find the values, we default use the previous one
    // to run properly
    printf(
        "MapSMtoCores for SM %d.%d is undefined."
        "  Default to use %d Cores/SM\n",
        major, minor, nGpuArchCoresPerSM[index - 1].Cores);
    return nGpuArchCoresPerSM[index - 1].Cores;
}