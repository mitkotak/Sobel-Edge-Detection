
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
#define NIMAGES 2048

// Block width WIDTH & HEIGHT
#define BLOCK_W 10
#define BLOCK_H 10

// buffer to read image into
// float image[HEIGHT][WIDTH];

// buffer for resulting image
// float final[HEIGHT][WIDTH];


#define MAXLINE 128

float total, sobel;
cudaEvent_t start_total, stop_total;
cudaEvent_t start_sobel, stop_sobel;

__global__ void warm_up_gpu(){
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  float ia, ib;
  ia = ib = 0.0f;
  ib += ia + tid; 
}

__global__ void imageBlur_horizontal(float *input, float *output, int width, int height) {

	int col = threadIdx.x + blockIdx.x * blockDim.x;
	int row = threadIdx.y + blockIdx.y * blockDim.y;

	int numcols = width;

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

	int numcols = width;

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

	int numcols = width;


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

	int numcols = width;

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

	int numcols = width;

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

	
	// pgmread("image100000x100000.pgm", (void *)image, WIDTH, HEIGHT);
	// pgmread("../images/image20000x20000.pgm", (void *)image, WIDTH, HEIGHT);
	// pgmread("../images/image16384x16384.pgm", (void *)image, WIDTH, HEIGHT);
	// pgmread("image10000x10000.pgm", (void *)image, WIDTH, HEIGHT);
	// pgmread("../images/image4096x4096.pgm", (void *)image, WIDTH, HEIGHT);
	// pgmread("../images/image2048x2048.pgm", (void *)image, WIDTH, HEIGHT);
	// pgmread("../images/rabbit2000x3000.pgm", (void *)image, WIDTH, HEIGHT);
	// pgmwrite("../images/image1024x1024.pgm", (void *)image, WIDTH, HEIGHT);
	// pgmread("../images/image512x512.pgm", (void *)image, WIDTH, HEIGHT);
	// pgmread("pgmimg.pgm", (void *)image, WIDTH, HEIGHT);

	// pgmwrite("image-outputl100000x100000.pgm", (void *)final, WIDTH, HEIGHT);
	// pgmwrite("image-output_ng_20000x20000.pgm", (void *)final, WIDTH, HEIGHT);
	// pgmwrite("../images/image-output_ng_16384x16384.pgm", (void *)final, WIDTH, HEIGHT);
	// pgmwrite("image-outputl10000x10000.pgm", (void *)final, WIDTH, HEIGHT);
	// pgmwrite("../images/image-output_g_4096x4096.pgm", (void *)final, WIDTH, HEIGHT);
	// pgmwrite("../images/image-output_ng_2048x2048.pgm", (void *)final, WIDTH, HEIGHT);
	// pgmwrite("../images/image-output_ng_2000x3000.pgm", (void *)final, WIDTH, HEIGHT);
	// pgmwrite("../images/image-output_ng_1024x1024.pgm", (void *)final, WIDTH, HEIGHT);
	// pgmwrite("../images/image-output_ng_512x512.pgm", (void *)final, WIDTH, HEIGHT);
	// // pgmwrite("pgmimg-output.pgm", (void *)final, WIDTH, HEIGHT);

int main(int argc, char *argv[])
{ 	int width = 600, height=600;

	size_t memSize = width * height * sizeof(float);
	float *image_array[NIMAGES];
	float *final_array[NIMAGES];
	float *d_input_array[NIMAGES];
	float *d_output_array[NIMAGES];
	float *gradient_h_output_array[NIMAGES];
	float *gradient_v_output_array[NIMAGES];

	for (int i=1; i <= NIMAGES; i++){
		float *image = NULL;
		checkCudaErrors((cudaMallocHost(&image, memSize)));
		pgmread("../images/test_images/apollonian_gasket.ascii.pgm", (void *)image, width, height);
		image_array[i] = image;
		
		float *final = NULL;
		checkCudaErrors((cudaMallocHost(&final, memSize)));
		final_array[i] = final;
		
		float *d_input, *d_output, *gradient_h_output, *gradient_v_output;

		checkCudaErrors(cudaMalloc(&d_input, memSize));
		checkCudaErrors(cudaMalloc(&d_output, memSize));
		checkCudaErrors(cudaMalloc(&gradient_h_output, memSize));
		checkCudaErrors(cudaMalloc(&gradient_v_output, memSize));

		d_input_array[i] = d_input;
		d_output_array[i] = d_output;
		gradient_h_output_array[i] = gradient_h_output;
		gradient_v_output_array[i] = gradient_v_output;

	}

  cudaEventCreate(&start_total);
  cudaEventCreate(&stop_total);
  cudaEventRecord(start_total, 0);


	printf("Block size: %dx%d\n", BLOCK_W, BLOCK_H);

	printf("Blocks per grid (width): %d |", (width / BLOCK_W));
	printf("Blocks per grid (height): %d \n", (height / BLOCK_H));
	
	dim3 threads(BLOCK_W, BLOCK_H); // threads per block
	dim3 blocks(width / BLOCK_W, height / BLOCK_H); // blocks per grid 

	warm_up_gpu << <blocks, threads >> > ();

	float avg_sobel = 0.0;

	for (int j=1; j <= 20; j++){
	
	cudaEventCreate(&start_sobel);
  	cudaEventCreate(&stop_sobel);

	cudaEventRecord(start_sobel, 0);

	for (int i=1; i<=NIMAGES; i++){

		float *image = image_array[i];
		float *d_input = d_input_array[i];
		float *d_output = d_output_array[i];
		float *gradient_h_output = gradient_h_output_array[i];
		float *gradient_v_output = gradient_v_output_array[i];
		float* final = final_array[i];

	cudaMemcpy(d_input, image, memSize, cudaMemcpyHostToDevice);

	// printf("Launching imageBlur_horizontal \n");
  	imageBlur_horizontal << <blocks, threads >> > (d_input, d_output, width, height);
	// printf("Launching imageBlur_vertical \n");
	imageBlur_vertical << <blocks, threads >> > (d_input, d_output, width, height);

	// printf("Copying data to device \n");
	// printf("Launching gradient_horizontal \n");
	gradient_horizontal<< <blocks, threads>> >(d_input, gradient_h_output, width, height);
	// printf("Launching gradient_vertical \n");
	gradient_vertical<< <blocks, threads>> >(d_input, gradient_v_output, width, height);
	// printf("Launching sobelFilter \n");	
	sobelFilter << <blocks, threads >> > (d_input, d_output, gradient_h_output, gradient_v_output, width, height);

	// printf("Copying data back to host \n");
	cudaMemcpy(final, d_output, memSize, cudaMemcpyDeviceToHost);

	}
	
	cudaThreadSynchronize();

	cudaEventRecord(stop_sobel, 0);
  	cudaEventSynchronize(stop_sobel);
  	cudaEventElapsedTime(&sobel, start_sobel, stop_sobel);
	
	 avg_sobel += sobel/20;

	}

	printf("Total Avg Device Time:  %f s \n", avg_sobel/1000);

	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err)
	{
		fprintf(stderr, "Cuda error: %s: %s.\n", "Main Loop", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	
	for (int i=1; i <= NIMAGES; i++){
		float *d_input = d_input_array[i];
		cudaFree(d_input);
		float *d_output = d_output_array[i];
		cudaFree(d_output);
		float *gradient_h_output = gradient_h_output_array[i];
		cudaFree(gradient_h_output);
		float *gradient_v_output = gradient_v_output_array[i];
		cudaFree(gradient_v_output);

		float* final = final_array[i];
		// write image
		pgmwrite("../images/test_images/image-output_g_apollonian_gasket.ascii.pgm", (void *)final,width, height);
	}
 
  cudaEventRecord(stop_total, 0);
  cudaEventSynchronize(stop_total);
  cudaEventElapsedTime(&total, start_total, stop_total);

  printf("Total Time:  %f s \n", total/1000);
    
	cudaDeviceReset();
	
	return 0;
}

