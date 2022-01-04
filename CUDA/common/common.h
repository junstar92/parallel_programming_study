#ifndef __COMMON_H__
#define __COMMON_H__

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#ifdef _WIN32
#include <Windows.h>
#include <stdint.h>
int gettimeofday(struct timeval * tp, struct timezone * tzp)
{
    // Note: some broken versions only have 8 trailing zero's, the correct epoch has 9 trailing zero's
    // This magic number is the number of 100 nanosecond intervals since January 1, 1601 (UTC)
    // until 00:00:00 January 1, 1970 
    static const uint64_t EPOCH = ((uint64_t) 116444736000000000ULL);

    SYSTEMTIME  system_time;
    FILETIME    file_time;
    uint64_t    time;

    GetSystemTime( &system_time );
    SystemTimeToFileTime( &system_time, &file_time );
    time =  ((uint64_t)file_time.dwLowDateTime )      ;
    time += ((uint64_t)file_time.dwHighDateTime) << 32;

    tp->tv_sec  = (long) ((time - EPOCH) / 10000000L);
    tp->tv_usec = (long) (system_time.wMilliseconds * 1000);

    return 0;
}
#else
#include <sys/time.h>
#endif

#define GET_TIME(now) { \
    struct timeval t; \
    gettimeofday(&t, NULL); \
    now = t.tv_sec + t.tv_usec/1000000.0; \
}

#define CUDA_CHECK(val) { \
	if (val != cudaSuccess) { \
		fprintf(stderr, "Error %s at line %d in file %s\n", cudaGetErrorString(val), __LINE__, __FILE__); \
		exit(1); \
	} \
}

#define COMPUTE_MATADD_PERFORMANCE(start, stop, M, N, threadsPerBlock) { \
    float msecTotal = 0.0f; \
    CUDA_CHECK(cudaEventElapsedTime(&msecTotal, start, stop)); \
    double flopsPerMatrixMul = static_cast<double>(M) * static_cast<double>(N); \
    double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecTotal / 1000.0f); \
    printf("Performance= %.2f GFlop/s, Time= %.3f msec, Size = %.0f Ops, " \
           "WorkgroupSize= %u threads/block\n",  \
           gigaFlops, msecTotal, flopsPerMatrixMul, threadsPerBlock); \
}

#define COMPUTE_MATMUL_PERFORMANCE(start, stop, M, K, N, threadsPerBlock) { \
    float msecTotal = 0.0f; \
    CUDA_CHECK(cudaEventElapsedTime(&msecTotal, start, stop)); \
    double flopsPerMatrixMul = 2.0 * static_cast<double>(M) * \
                                static_cast<double>(K) * static_cast<double>(N); \
    double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecTotal / 1000.0f); \
    printf("Performance= %.2f GFlop/s, Time= %.3f msec, Size = %.0f Ops, " \
           "WorkgroupSize= %u threads/block\n",  \
           gigaFlops, msecTotal, flopsPerMatrixMul, threadsPerBlock); \
}

template<typename T>
inline void common_random_init_vector(T *vec, int n)
{
    for (int i = 0; i < n; i++) {
        vec[i] = rand() / (T)RAND_MAX;
    }
}

template<typename T>
inline void common_random_init_matrix(T *mat, int m, int n)
{
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            mat[i*n + j] = rand() / (T)RAND_MAX;
        }
    }
}

template<typename T>
inline int common_verify_matAdd(T *A, T *B, T *C, int M, int N)
{
    printf("Verifying all results...\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            T val = A[i*N + j] + B[i*N + j];

            if (fabs(val - C[i*N + j]) > 1e-5) {
                printf("%f != %f\n", C[i*N + j], val);
                fprintf(stderr, "Result verification failed at (%d, %d)\n", i, j);
                fprintf(stderr, "Test FAILED\n");
                return -1;
            }
        }
    }
    printf(".....\n");
    printf("Test PASSED\n");

    return 0;
}

template<typename T>
inline int common_verify_matMul(T *A, T *B, T *C, int M, int K, int N)
{
    printf("Verifying all results...\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            T val = 0.0;
            for (int k = 0; k < K; k++) {
                val += A[i*K + k] * B[k*N + j];
            }

            if (fabs(val - C[i*N + j]) > 1e-5) {
                printf("%f != %f\n", C[i*N + j], val);
                fprintf(stderr, "Result verification failed at (%d, %d)\n", i, j);
                fprintf(stderr, "Test FAILED\n");
                return -1;
            }
        }
    }
    printf(".....\n");
    printf("Test PASSED\n");

    return 0;
}

template<typename T>
inline int common_verify_matMul_sampling(T *A, T *B, T *C, int M, int K, int N, int sampleCnt = 10)
{
    printf("Random Sampling Verifying...\n");
    for (int i = 0; i < sampleCnt; i++) {
        int idx = rand() % (M * N);
        int row = idx/N;
        int col = idx%N;

        printf("[INDEX (%d, %d)] checked... ", row, col);
        T val = 0.0;
        for (int j = 0; j < K; j++) {
            val += A[row*K + j] * B[N*j + col];
        }
        if (fabs(val - C[idx]) > 1e-5) {
            printf("%f != %f\n", C[idx], val);
            fprintf(stderr, "Result verification failed at (%d, %d)\n", row, col);
            fprintf(stderr, "Test FAILED\n");
            return -1;
        }
        else {
            printf("%f == %f\n", C[idx], val);
        }
    }
    printf(".....\n");
    printf("Test PASSED\n");

    return 0;
}

template<typename T>
int common_verify_matMul_l2ne(T *A, T *B, T *C, int M, int K, int N)
{
    printf("Verifying matrix multiplication by l2-norm error\n");
    const T epsilon = 1e-6;
    T error = 0;
    T ref = 0;

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            T val = 0.0;
            for (int k = 0; k < K; k++) {
                val += A[i*K + k] * B[k*N + j];
            }
            T diff = val - C[i*N + j];
            error += diff * diff;
            ref += val * val;
        }
    }

    float normRef = sqrtf(static_cast<float>(ref));
    if (fabs(ref) < 1e-7) {
        fprintf(stderr, "ERROR, reference l2-norm is 0\n");
        return -1;
    }

    float normError = sqrtf(static_cast<float>(error));
    float err = normError / (M*N);
    if (err > epsilon) {
        fprintf(stderr, "ERROR, l2-norm error %f is greater than epsilon %f\n", err, epsilon);
        fprintf(stderr, "Test FAILED\n");
        return -1;
    }

    printf(".....\n");
    printf("l2-norm error = %.9f\n", err);
    printf("Test PASSED\n");

    return 0;
}

#endif