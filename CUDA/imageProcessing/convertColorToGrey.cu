/*****************************************************************************
 * File:        convertColorToGrey.cu
 * Description: Convert color scale to grey scale of input image.
 *              This program doesn't save result image, and just show the result.
 *              For reading image, OpenCV library should be used.
 *              
 * Compile:     nvcc -o convertColorToGrey convertColorToGrey.cu -I.. -lcuda $(pkg-config opencv4 --libs --cflags)
 * Run:         ./convertColorToGrey <image file path>
 *****************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <common/common.h>

#include <cuda_runtime.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#define CHANNELS 3

void Usage(char prog_name[]);
__global__
void colorToGreyscaleConversion(unsigned char* in, unsigned char* out, int width, int height);

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
    h_resultImg = (unsigned char*)malloc(width * height * sizeof(unsigned char));
    (void)memcpy(h_origImg, origImg.data, width * height * channels);

    unsigned char *d_origImg, *d_resultImg;
    CUDA_CHECK(cudaMalloc((void**)&d_origImg, width * height * channels * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc((void**)&d_resultImg, width * height * sizeof(unsigned char)));

    // Copy the host input in host memory to the device input in device memory
    CUDA_CHECK(cudaMemcpy(d_origImg, h_origImg, width * height * channels * sizeof(unsigned char), cudaMemcpyHostToDevice));

    // Launch the Kernel
    const int block_size = 16;
    dim3 threads(block_size, block_size);
    dim3 grid(ceil(width / (double)threads.x), ceil(height / (double)threads.y));
    colorToGreyscaleConversion<<<grid, threads>>>(d_origImg, d_resultImg, width, height);
    
    // Copy the device result in device memory to the host result in host memory
    CUDA_CHECK(cudaMemcpy(h_resultImg, d_resultImg, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost));
    
    cv::Mat resultImg(height, width, CV_8UC1);
    memcpy(resultImg.data, h_resultImg, width * height);

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

// Input image has 3 channels corresponding to RGB
// The input image is encoded as unsigned characters [0, 255]
__global__
void colorToGreyscaleConversion(unsigned char* in, unsigned char* out, int width, int height)
{
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    if (Row < height && Col < width) {
        int offset = Row*width + Col;
        int rgbOffset = offset*CHANNELS;
        
        unsigned char r = in[rgbOffset];       // red value for pixel
        unsigned char g = in[rgbOffset + 1];   // green value for pixel
        unsigned char b = in[rgbOffset + 2];   // blue value for pixel

        out[offset] = 0.21f * r + 0.71f * g + 0.07f * b;
    }
}