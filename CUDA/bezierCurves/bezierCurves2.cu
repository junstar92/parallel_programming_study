/*****************************************************************************
 * File:        bezierCurves2.cu
 * Description: Implement Bezier Curve Calculation with dynamic parallelism
 *              
 * Compile:     nvcc -arch=sm_70 -rdc=true -o bezierCurves2 bezierCurves2.cu -I.. -I. $(pkg-config opencv4 --libs --cflags) [-DUSE_STREAM]
 * Run:         ./bezierCurves2
 * Argument:    n.a
 *****************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <common/common.h>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <cuda.h>
#include "bezierCurves.cuh"

__global__ void computeBezierLine_child(int lineIdx, BezierLine* bLines, int nTessPoints)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x; // Compute idx unique to this vertex
    
    if (idx < nTessPoints) {
        float u = (float)idx / (float)(nTessPoints - 1); // Compute u from idx
        float omu = 1.0f - u;                            // pre-compute one minus u
        float B3u[3];                                    // Compute quadratic Bezier coefficients
        B3u[0] = omu * omu;
        B3u[1] = 2.0f * u * omu;
        B3u[2] = u * u;
        float2 position = {0, 0}; // Set position to zero
        for (int i = 0; i < 3; i++) {
            // Add the contribution of the i'th control point to position
            position = position + B3u[i] * bLines[lineIdx].CP[i];
        }
        // Assign value of vertex position to the correct array element
        bLines[lineIdx].vertexPos[idx] = position;
    }
}

__global__ void computeBezierLines_parent(BezierLine* bLines, int nLines)
{
    // Compute a unique index for each Bezier line
    int lineIdx = blockDim.x * blockIdx.x + threadIdx.x;
    if (lineIdx < nLines) {
        // Compute the curvature of the line
        float curvature = computeCurvature(bLines);

#ifdef USE_STREAM
        // Create non-blocking stream
        cudaStream_t stream;
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
#endif

        // From the curvature, compute the number of tessellation points
        bLines[lineIdx].nVertices = min(max((int)(curvature*16.0f), 4), 32);

        // Call the child kernel to compute the tessellated points for each line
#ifdef USE_STREAM
        computeBezierLine_child<<<ceil((float)bLines[lineIdx].nVertices/23.0f), 32, 0, stream>>>
                (lineIdx, bLines, bLines[lineIdx].nVertices);
        // Destroy stream
        cudaStreamDestroy(stream);
#else
        computeBezierLine_child<<<ceil((float)bLines[lineIdx].nVertices/23.0f), 32>>>
                (lineIdx, bLines, bLines[lineIdx].nVertices);
#endif
    }
}

// Main function
int main(int argc, char **argv)
{
    CUDA_CHECK(cudaSetDevice(0));

    BezierLine *bLines_h = new BezierLine[N_LINES];
    initializeBLines(bLines_h);

    BezierLine *bLines_d;
    CUDA_CHECK(cudaMalloc((void **)&bLines_d, N_LINES * sizeof(BezierLine)));
    CUDA_CHECK(cudaMemcpy(bLines_d, bLines_h, N_LINES * sizeof(BezierLine), cudaMemcpyHostToDevice));

    double start, finish;
    GET_TIME(start);
    computeBezierLines_parent<<<ceil((float)N_LINES/BLOCK_DIM), BLOCK_DIM>>>(bLines_d, N_LINES);
    CUDA_CHECK(cudaMemcpy(bLines_h, bLines_d, N_LINES*sizeof(BezierLine), cudaMemcpyDeviceToHost));
    GET_TIME(finish);
    
    printf("Elapsed time: %.6f msec\n", (finish - start)*1000);

    const int rows = 4;
    const int cols = 4;
    const int img_width = 196;
    cv::Mat dstImage(img_width * (rows + 1), img_width * (cols + 1), CV_8UC3, cv::Scalar(255, 255, 255));

    int max_points = 0;
    const int numberOfdisplay = 16;
    for (int i = 0; i < numberOfdisplay; i++) {
        const int r = i / cols;
        const int c = i % cols;
        for (int j = 0; j < 2; j++) {
            cv::line(dstImage,
                    cv::Point((r*img_width) + ((img_width/4) + bLines_h[i].CP[j].x*img_width), (c*img_width) + ((img_width/4) + bLines_h[i].CP[j].y*img_width)),
                    cv::Point((r*img_width) + ((img_width/4) + bLines_h[i].CP[j+1].x*img_width), (c*img_width) + ((img_width/4) + bLines_h[i].CP[j+1].y*img_width)),
                    cv::Scalar(0,0,0), 2);
        }

        if (bLines_h[i].nVertices > max_points)
            max_points = bLines_h[i].nVertices;
    }


    for (int k = 0; k < max_points - 1; k++) {
        for (int i = 0; i < numberOfdisplay; i++) {
            const int r = i / cols;
            const int c = i % cols;

            if (k < bLines_h[i].nVertices - 1) {
                cv::line(dstImage,
                        cv::Point(r*img_width + ((img_width/4) + bLines_h[i].vertexPos[k].x*img_width), c*img_width + ((img_width/4) + bLines_h[i].vertexPos[k].y*img_width)),
                        cv::Point(r*img_width + ((img_width/4) + bLines_h[i].vertexPos[k+1].x*img_width), c*img_width + ((img_width/4) + bLines_h[i].vertexPos[k+1].y*img_width)),
                        cv::Scalar(255,0,0), 2);
            };
        }
        cv::imshow("win", dstImage);
        cv::waitKey(500);
    }
    cv::waitKey(0);

    CUDA_CHECK(cudaFree(bLines_d));
    delete[] bLines_h;

    return 0;
}