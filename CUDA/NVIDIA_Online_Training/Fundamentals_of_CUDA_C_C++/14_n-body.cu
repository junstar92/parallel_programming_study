#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "timer.h"
#include "files.h"

#define SOFTENING 1e-9f

/*
 * Each body contains x, y, and z coordinate positions,
 * as well as velocities in the x, y, and z directions.
 */

typedef struct { float x, y, z, vx, vy, vz; } Body;

/*
 * Calculate the gravitational impact of all bodies in the system
 * on all others.
 */

__global__
void bodyForce(Body *p, float dt, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < n; i += stride) {
        float Fx = 0.f, Fy = 0.f, Fz = 0.f;
        
        for (int j = 0; j < n; j++) {
            float dx = p[j].x - p[i].x;
            float dy = p[j].y - p[i].y;
            float dz = p[j].z - p[i].z;
            float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
            float invDist = rsqrtf(distSqr);
            float invDist3 = invDist * invDist * invDist;
            
            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }
        
        p[i].vx += dt * Fx;
        p[i].vy += dt * Fy;
        p[i].vz += dt * Fz;
    }
}

__global__
void intergratePosition(Body *p, float dt, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx ; i < n; i += stride) 
    {
        p[i].x += p[i].vx * dt;
        p[i].y += p[i].vy * dt;
        p[i].z += p[i].vz * dt;
    }
}

int main(const int argc, const char **argv)
{
    int nBodies = 2 << 11;
    if (argc > 1)
        nBodies = 2 << atoi(argv[1]);

    int deviceId;
    checkCuda(cudaGetDevice(&deviceId));
    cudaDeviceProp props;
    checkCuda(cudaGetDeviceProperties(&props, deviceId));

    size_t threadsPerBlock = props.maxThreadsPerBlock;
    size_t numberOfBlocks = props.multiProcessorCount;

    // The assessment will pass hidden initialized values to check for correctness.
    // You should not make changes to these files, or else the assessment will not work.
    const char *initialized_values;
    const char *solution_values;

    if (nBodies == 2 << 11)
    {
        initialized_values = "09-nbody/files/initialized_4096";
        solution_values = "09-nbody/files/solution_4096";
    }
    else
    { // nBodies == 2<<15
        initialized_values = "09-nbody/files/initialized_65536";
        solution_values = "09-nbody/files/solution_65536";
    }

    if (argc > 2)
        initialized_values = argv[2];
    if (argc > 3)
        solution_values = argv[3];

    const float dt = 0.01f; // Time step
    const int nIters = 10;  // Simulation iterations

    int bytes = nBodies * sizeof(Body);
    float *buf;

    cudaMallocManaged(&buf, bytes);

    Body *p = (Body *)buf;

    cudaMemPrefetchAsync(buf, bytes, cudaCpuDeviceId);
    read_values_from_file(initialized_values, buf, bytes);

    double totalTime = 0.0;

    /*
     * This simulation will run for 10 cycles of time, calculating gravitational
     * interaction amongst bodies, and adjusting their positions to reflect.
     */
    cudaMemPrefetchAsync(buf, bytes, device_id);

    for (int iter = 0; iter < nIters; iter++)
    {
        StartTimer();

        bodyForce<<<numberOfBlocks, threadsPerBlock>>>(p, dt, nBodies); // compute interbody forces
        intergratePosition<<<numberOfBlocks, threadsPerBlock>>>(p, dt, nBodies);

        cudaDeviceSynchronize();

        const double tElapsed = GetTimer() / 1000.0;
        totalTime += tElapsed;
    }

    double avgTime = totalTime / (double)(nIters);
    float billionsOfOpsPerSecond = 1e-9 * nBodies * nBodies / avgTime;

    cudaMemPrefetchAsync(buf, bytes, cudaCpuDeviceId);
    write_values_to_file(solution_values, buf, bytes);

    // You will likely enjoy watching this value grow as you accelerate the application,
    // but beware that a failure to correctly synchronize the device might result in
    // unrealistically high values.
    printf("%0.3f Billion Interactions / second\n", billionsOfOpsPerSecond);

    cudaFree(buf);
}