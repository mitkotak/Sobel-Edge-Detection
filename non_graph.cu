
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifdef __INTELLISENSE__
void __syncthreads();
#endif

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <helper_cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>
#include <pgmio.h>
#include <vector>

// image dimensions WIDTH & HEIGHT
#define WIDTH  256
#define HEIGHT 256

// Block width WIDTH & HEIGHT
#define BLOCK_W 32
#define BLOCK_H 32

// buffer to read image into
// float image[HEIGHT][WIDTH];

// buffer for resulting image
// float final[HEIGHT][WIDTH];

// prototype declarations

void load_image(float *image);
void call_kernel(float *image, float *final);
void save_image(float *final);

#define MAXLINE 128

float total, sobel;
cudaEvent_t start_total, stop_total;
cudaEvent_t start_sobel, stop_sobel;

__global__ void imageBlur_horizontal(float *input, float *output, int width, int height) {

	int col = threadIdx.x + blockIdx.x * blockDim.x;
	int row = threadIdx.y + blockIdx.y * blockDim.y;

	int numcols = WIDTH;

	float blur;

	if (row <= height && col <= width && row > 0 && col > 0)
	{
		// weights
		int	x3, x4, x5;

		// blur
		// 0.2 0.2 0.2

		x3 = input[row * numcols + (col - 1)];			// left
		x4 = input[row * numcols + col];				// center
		x5 = input[row * numcols + (col + 1)];			// right

		blur =  (x3 * 0.2) + (x4 * 0.2) + (x5 * 0.2);

		output[row * numcols + col] = blur;
	}
}

__global__ void imageBlur_vertical(float *input, float *output, int width, int height) {
	
	int col = threadIdx.x + blockIdx.x * blockDim.x;
	int row = threadIdx.y + blockIdx.y * blockDim.y;

	int numcols = WIDTH;

	float blur;

	if (row <= height && col <= width && row > 0 && col > 0)
	{
		// weights
		int		x1,x7;

		// blur
		// 0.0 0.2 0.0
		// 0.2 0.2 0.2
		// 0.0 0.2 0.0

		x1 = input[(row + 1) * numcols + col];			// up
		x7 = input[(row + -1) * numcols + col];			// down

		blur = (x1 * 0.2) + (x7 * 0.2);

		output[row * numcols + col] = blur;
	}
}


__global__ void gradient_horizontal(float *input, float *output, int width, int height) {

	int col = threadIdx.x + blockIdx.x * blockDim.x;
	int row = threadIdx.y + blockIdx.y * blockDim.y;

	int numcols = WIDTH;


	// horizontal
	// -1  0  1
	// -2  0  2
	// -1  0  1

	int x0, x2, 
		x3, x5, 
		x6, x8;

	if (row <= height && col <= width && row > 0 && col > 0)
	{
	
	x0 = input[(row - 1) * numcols + (col - 1)];	// leftup
	x2 = input[(row - 1) * numcols + (col + 1)];	// rightup
	x3 = input[row * numcols + (col - 1)];			// left
	x5 = input[row * numcols + (col + 1)];			// right
	x6 = input[(row + 1) * numcols + (col - 1)];	// leftdown
	x8 = input[(row + 1) * numcols + (col + 1)];	// rightdown


	output[row * numcols + col] = (x0 * -1) + (x2 * 1) + (x3 * -2) + (x5 * 2) + (x6 * -1) + (x8 * 1);

	}

	return;
}


__global__ void gradient_vertical(float *input, float *output, int width, int height) {

	int col = threadIdx.x + blockIdx.x * blockDim.x;
	int row = threadIdx.y + blockIdx.y * blockDim.y;

	int numcols = WIDTH;

	// vertical 
	// -1 -2 -1
	//  0  0  0
	//  1  2  1

	int x0, x1, 
		x3, x6, 
		x7, x8;

	if (row <= height && col <= width && row > 0 && col > 0)
	{
		x0 = input[(row - 1) * numcols + (col - 1)];	// leftup
		x1 = input[(row + 1) * numcols + col];			// up
		x3 = input[row * numcols + (col - 1)];			// left
		x6 = input[(row + 1) * numcols + (col - 1)];	// leftdown
		x7 = input[(row + -1) * numcols + col];			// down
		x8 = input[(row + 1) * numcols + (col + 1)];	// rightdown


		output[row * numcols + col] = (x0 * -1) + (x1 * -2) + (x3 * -1) + (x6 * 1) + (x7 * 2) + (x8 * 1);

	}
	
}

__global__ void sobelFilter(float *input, float *output, float *gradient_h_output, float *gradient_v_output, int width, int height) {
	
	int col = threadIdx.x + blockIdx.x * blockDim.x;
	int row = threadIdx.y + blockIdx.y * blockDim.y;

	int numcols = WIDTH;

	float gradient,gradient_h,gradient_v;	
	float thresh = 30;	

	if (row <= height && col <= width && row > 0 && col > 0)
	{	

		gradient_h = gradient_h_output[row * numcols + col];
		gradient_v = gradient_v_output[row * numcols + col];
		gradient = sqrt((gradient_h * gradient_h) + (gradient_v * gradient_v));

		if (gradient >= thresh)
		{
			gradient = 255;
		}
		else {
			gradient = 0;
		}
		output[row * numcols + col] = gradient;
	}
}

void load_image(float *image) {
	// pgmread("image100000x100000.pgm", (void *)image, WIDTH, HEIGHT);
	// pgmread("image16384x16384.pgm", (void *)image, WIDTH, HEIGHT);
	// pgmread("image10000x10000.pgm", (void *)image, WIDTH, HEIGHT);
	// pgmread("image4096x4096.pgm", (void *)image, WIDTH, HEIGHT);
	// pgmread("../images/rabbit2000x3000.pgm", (void *)image, WIDTH, HEIGHT);
	// pgmwrite("image1024x1024.pgm", (void *)image, WIDTH, HEIGHT);
	pgmread("../images/image512x512.pgm", (void *)image, WIDTH, HEIGHT);
	// pgmread("pgmimg.pgm", (void *)image, WIDTH, HEIGHT);
}

void save_image(float *final) {
	// pgmwrite("image-outputl100000x100000.pgm", (void *)final, WIDTH, HEIGHT);
	// pgmwrite("image-outputl16384x16384.pgm", (void *)final, WIDTH, HEIGHT);
	// pgmwrite("image-outputl10000x10000.pgm", (void *)final, WIDTH, HEIGHT);
	// pgmwrite("image-outputl4096x4096.pgm", (void *)final, WIDTH, HEIGHT);
	// pgmwrite("../images/image-output_ng_2000x3000.pgm", (void *)final, WIDTH, HEIGHT);
	// pgmwrite("image-outputl1024x1024.pgm", (void *)final, WIDTH, HEIGHT);
	pgmwrite("image-output_ng_512x512.pgm", (void *)final, WIDTH, HEIGHT);
	// pgmwrite("pgmimg-output.pgm", (void *)final, WIDTH, HEIGHT);
}

void call_kernel(float *image, float *final) {
	int width = WIDTH, height=HEIGHT;
	int x, y;
	float *d_input, *d_output, *gradient_h_output, *gradient_v_output;

	printf("Block size: %dx%d\n", BLOCK_W, BLOCK_H);

	float memSize = WIDTH * HEIGHT * sizeof(float);

	cudaMalloc(&d_input, memSize);
	cudaMalloc(&d_output, memSize);
	cudaMalloc(&gradient_h_output, memSize);
	cudaMalloc(&gradient_v_output, memSize);

	printf("Blocks per grid (width): %d |", (WIDTH / BLOCK_W));
	printf("Blocks per grid (height): %d \n", (HEIGHT / BLOCK_H));

	cudaMemcpy(d_input, image, memSize, cudaMemcpyHostToDevice);

	dim3 threads(BLOCK_W, BLOCK_H); // threads per block
	dim3 blocks(WIDTH / BLOCK_W, HEIGHT / BLOCK_H); // blocks per grid 
	
	// printf("Launching imageBlur_horizontal \n");
  	imageBlur_horizontal << <blocks, threads >> > (d_input, d_output, WIDTH, HEIGHT);
	// printf("Launching imageBlur_vertical \n");
	imageBlur_vertical << <blocks, threads >> > (d_input, d_output, WIDTH, HEIGHT);
  
  	cudaThreadSynchronize();
	// printf("Copying data to device \n");
  	cudaMemcpy(d_input, d_output, memSize, cudaMemcpyDeviceToHost);
	// printf("Launching gradient_horizontal \n");
	gradient_horizontal<< <blocks, threads>> >(d_input, gradient_h_output, WIDTH, HEIGHT);
	// printf("Launching gradient_vertical \n");
	gradient_vertical<< <blocks, threads>> >(d_input, gradient_v_output, WIDTH, HEIGHT);
	// printf("Launching sobelFilter \n");	
	sobelFilter << <blocks, threads >> > (d_input, d_output, gradient_h_output, gradient_v_output, WIDTH, HEIGHT);

	cudaThreadSynchronize();
	// printf("Copying data back to host \n");
	cudaMemcpy(final, d_output, memSize, cudaMemcpyDeviceToHost);

	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err)
	{
		fprintf(stderr, "Cuda error: %s: %s.\n", "Main Loop", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	cudaFree(d_input);
	cudaFree(d_input);
	cudaFree(d_output);
	cudaFree(gradient_h_output);
	cudaFree(gradient_v_output);
}

int main(int argc, char *argv[])
{
  float *image = NULL, *final = NULL;
  size_t memSize = WIDTH * HEIGHT * sizeof(float);
  checkCudaErrors((cudaMallocHost(&image, memSize)));
  checkCudaErrors((cudaMallocHost(&final, memSize)));

  cudaEventCreate(&start_total);
  cudaEventCreate(&stop_total);
    
  cudaEventCreate(&start_sobel);
  cudaEventCreate(&stop_sobel);
    
  cudaEventRecord(start_total, 0);

  load_image(image);
   
  cudaEventRecord(start_sobel, 0);

  call_kernel(image,final);
  
  cudaEventRecord(stop_sobel, 0);
  cudaEventSynchronize(stop_sobel);
  cudaEventElapsedTime(&sobel, start_sobel, stop_sobel);

  save_image(final);
   
  cudaEventRecord(stop_total, 0);
  cudaEventSynchronize(stop_total);
  cudaEventElapsedTime(&total, start_total, stop_total);
    
  printf("Total Parallel Time:  %f s \n", sobel/1000);
  printf("Total Serial Time:  %f s \n", (total-sobel)/1000);
  printf("Total Time:  %f s \n", total/1000);
  
    
	cudaDeviceReset();
	
	return 0;
}

