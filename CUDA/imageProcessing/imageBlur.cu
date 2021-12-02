/*****************************************************************************
 * File:        imageBlur.cu
 * Description: Blur input image using 3D blocks.
 *              This program doesn't save result image, and just show the result.
 *              For reading image, OpenCV library should be used.
 *              
 * Compile:     nvcc -o imageBlur imageBlur.cu -I.. -lcuda $(pkg-config opencv4 --libs --cflags)
 * Run:         ./imageBlur <image file path>
 *****************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <common/common.h>

#include <cuda_runtime.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#define CHANNELS 3
#define BLUR_SIZE 10

void Usage(char prog_name[]);
__global__
void blurKernel(unsigned char* in, unsigned char* out, int width, int height, int channel);

int main(int argc, char** argv)
{
    if (argc != 2) {
        Usage(argv[0]);
    }
    
    const char* file_name = argv[1];
    int width, height, channels;
    unsigned char *h_origImg, *h_resultImg;
    // open image file
    cv::Mat origImg = cv::imread(file_name);

    width = origImg.cols;
    height = origImg.rows;
    channels = origImg.channels();
    printf("Image size = (%d x %d x %d)\n", width, height, channels);
    assert(channels == CHANNELS);
    
    cv::Mat half;
    cv::resize(origImg, half, cv::Size(width/2, height/2));
    cv::imshow("image", half);
    cv::waitKey(0);

    h_origImg = (unsigned char*)malloc(width * height * channels * sizeof(unsigned char));
    h_resultImg = (unsigned char*)malloc(width * height * channels * sizeof(unsigned char));
    (void)memcpy(h_origImg, origImg.data, width * height * channels);

    unsigned char *d_origImg, *d_resultImg;
    CUDA_CHECK(cudaMalloc((void**)&d_origImg, width * height * channels * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc((void**)&d_resultImg, width * height * channels * sizeof(unsigned char)));

    // Copy the host input in host memory to the device input in device memory
    CUDA_CHECK(cudaMemcpy(d_origImg, h_origImg, width * height * channels * sizeof(unsigned char), cudaMemcpyHostToDevice));

    // Launch the blur Kernel
    const int block_size = 16;
    dim3 threads(block_size, block_size, channels);
    dim3 grid(ceil(height / (double)threads.x), ceil(width / (double)threads.y));
    blurKernel<<<grid, threads>>>(d_origImg, d_resultImg, width, height, channels);
    
    // Copy the device result in device memory to the host result in host memory
    CUDA_CHECK(cudaMemcpy(h_resultImg, d_resultImg, width * height * channels * sizeof(unsigned char), cudaMemcpyDeviceToHost));
    
    cv::Mat resultImg(height, width, CV_8UC3);
    memcpy(resultImg.data, h_resultImg, width * height * channels);

    // Free device global memory
    CUDA_CHECK(cudaFree(d_origImg));
    CUDA_CHECK(cudaFree(d_resultImg));

    // Free host memory
    free(h_origImg);
    free(h_resultImg);

    // show result
    //cv::Mat resizeImg;
    cv::resize(resultImg, resultImg, cv::Size(width/2, height/2));
    cv::imshow("image", resultImg);
    cv::waitKey(0);

    return 0;
}

void Usage(char prog_name[])
{
    fprintf(stderr, "Usage: %s <image file path>\n", prog_name);
    exit(EXIT_FAILURE);
}

__global__
void blurKernel(unsigned char* in, unsigned char* out, int width, int height, int channel)
{
    int Plane = blockIdx.z * blockDim.z + threadIdx.z;
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    if (Row < height && Col < width && Plane < channel) {
        int pixelVal = 0;
        int pixelCnt = 0;

        for (int bRow = -BLUR_SIZE; bRow < BLUR_SIZE; bRow++) {
            for (int bCol = -BLUR_SIZE; bCol < BLUR_SIZE; bCol++) {
                int curRow = Row + bRow;
                int curCol = Col + bCol;

                if (curRow >= 0 && curRow < height && curCol >= 0 && curCol < width) {
                    pixelVal += in[(curRow * width + curCol) * channel + Plane];
                    pixelCnt++;
                }
            }
        }

        out[(Row * width + Col) * channel + Plane] = (unsigned char)(pixelVal / pixelCnt);
    }
}