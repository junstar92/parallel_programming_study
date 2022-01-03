#include <stdio.h>
#include <cuda.h>

#define MAX_TESS_POINTS 32
#define N_LINES 256
#define BLOCK_DIM 32

// A structure containing all paramters needed to tessellate a Bezier line
struct BezierLine
{
    float2 CP[3];                       // Control Points for the line
    float2 vertexPos[MAX_TESS_POINTS];  // Vertex position array to tessellate into
    int nVertices;                      // Number of tessellated vertices
};

__forceinline__ __device__ float2 operator+(float2 a, float2 b)
{
    float2 c;
    c.x = a.x + b.x;
    c.y = a.y + b.y;
    return c;
}

__forceinline__ __device__ float2 operator-(float2 a, float2 b)
{
    float2 c;
    c.x = a.x - b.x;
    c.y = a.y - b.y;
    return c;
}

__forceinline__ __device__ float2 operator*(float a, float2 b)
{
    float2 c;
    c.x = a * b.x;
    c.y = a * b.y;
    return c;
}

__forceinline__ __device__ float length(float2 a)
{
    return sqrtf((a.x * a.x) + (a.y * a.y));
}

__forceinline__ __device__ float computeCurvature(BezierLine *bLines)
{
    int bIdx = blockIdx.x;
    float curvature = length(bLines[bIdx].CP[1] - 0.5f * (bLines[bIdx].CP[0] + bLines[bIdx].CP[2])) 
                / length(bLines[bIdx].CP[2] - bLines[bIdx].CP[0]);

    return curvature;
}

void initializeBLines(BezierLine *bLines_h)
{
    float2 last = {0, 0};
    for (int i = 0; i < N_LINES; i++)
    {
        // Set first point of this line to last point of previous line
        bLines_h[i].CP[0] = last;
        for (int j = 1; j < 3; j++)
        {
            // Assign random corrdinate between 0 and 1
            bLines_h[i].CP[j].x = (float)rand() / RAND_MAX;
            bLines_h[i].CP[j].y = (float)rand() / RAND_MAX;
        }
        last = bLines_h[i].CP[2]; // keep the last point of this line
        // Set numbeer of tessellated vertices to zero
        bLines_h[i].nVertices = 0;
    }
}