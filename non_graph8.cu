
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
#define WIDTH 600
#define HEIGHT 600

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
{ int width = 600, height=600;
  float *image = NULL, *final = NULL;
  float *image2 = NULL, *final2 = NULL;
  float *image3 = NULL, *final3 = NULL;
  float *image4 = NULL, *final4 = NULL;
  float *image5 = NULL, *final5 = NULL;
  float *image6 = NULL, *final6 = NULL;
  float *image7 = NULL, *final7 = NULL;
  float *image8 = NULL, *final8 = NULL;

  size_t memSize = width * height * sizeof(float);
  checkCudaErrors((cudaMallocHost(&image, memSize)));
  checkCudaErrors((cudaMallocHost(&final, memSize)));

 checkCudaErrors((cudaMallocHost(&image2, memSize)));
  checkCudaErrors((cudaMallocHost(&final2, memSize)));

checkCudaErrors((cudaMallocHost(&image3, memSize)));
  checkCudaErrors((cudaMallocHost(&final3, memSize)));

 checkCudaErrors((cudaMallocHost(&image4, memSize)));
  checkCudaErrors((cudaMallocHost(&final4, memSize)));

  checkCudaErrors((cudaMallocHost(&image5, memSize)));
  checkCudaErrors((cudaMallocHost(&final5, memSize)));

 checkCudaErrors((cudaMallocHost(&image6, memSize)));
  checkCudaErrors((cudaMallocHost(&final6, memSize)));

checkCudaErrors((cudaMallocHost(&image7, memSize)));
  checkCudaErrors((cudaMallocHost(&final7, memSize)));

 checkCudaErrors((cudaMallocHost(&image8, memSize)));
  checkCudaErrors((cudaMallocHost(&final8, memSize)));

  // read image 
  pgmread("../images/test_images/apollonian_gasket.ascii.pgm", (void *)image, width, height);
  pgmread("../images/test_images/apollonian_gasket.ascii.pgm", (void *)image2, width, height);
  pgmread("../images/test_images/apollonian_gasket.ascii.pgm", (void *)image3, width, height);
  pgmread("../images/test_images/apollonian_gasket.ascii.pgm", (void *)image4, width, height);
   pgmread("../images/test_images/apollonian_gasket.ascii.pgm", (void *)image5, width, height);
  pgmread("../images/test_images/apollonian_gasket.ascii.pgm", (void *)image6, width, height);
  pgmread("../images/test_images/apollonian_gasket.ascii.pgm", (void *)image7, width, height);
  pgmread("../images/test_images/apollonian_gasket.ascii.pgm", (void *)image8, width, height);
  
  cudaEventCreate(&start_total);
  cudaEventCreate(&start_total);
  cudaEventCreate(&start_total);
  cudaEventCreate(&start_total);
  cudaEventCreate(&stop_total);
  cudaEventRecord(start_total, 0);

	int x, y;
	float *d_input, *d_output, *gradient_h_output, *gradient_v_output;
	float *d_input2, *d_output2, *gradient_h_output2, *gradient_v_output2;
	float *d_input3, *d_output3, *gradient_h_output3, *gradient_v_output3;
	float *d_input4, *d_output4, *gradient_h_output4, *gradient_v_output4;
	float *d_input5, *d_output5, *gradient_h_output5, *gradient_v_output5;
	float *d_input6, *d_output6, *gradient_h_output6, *gradient_v_output6;
	float *d_input7, *d_output7, *gradient_h_output7, *gradient_v_output7;
	float *d_input8, *d_output8, *gradient_h_output8, *gradient_v_output8;


	printf("Block size: %dx%d\n", BLOCK_W, BLOCK_H);

	printf("Blocks per grid (width): %d |", (width / BLOCK_W));
	printf("Blocks per grid (height): %d \n", (height / BLOCK_H));
	
	dim3 threads(BLOCK_W, BLOCK_H); // threads per block
	dim3 blocks(width / BLOCK_W, height / BLOCK_H); // blocks per grid 

	warm_up_gpu << <blocks, threads >> > ();

	cudaMalloc(&d_input, memSize);
	cudaMalloc(&d_output, memSize);
	cudaMalloc(&gradient_h_output, memSize);
	cudaMalloc(&gradient_v_output, memSize);

	cudaMalloc(&d_input2, memSize);
	cudaMalloc(&d_output2, memSize);
	cudaMalloc(&gradient_h_output2, memSize);
	cudaMalloc(&gradient_v_output2, memSize);

	cudaMalloc(&d_input3, memSize);
	cudaMalloc(&d_output3, memSize);
	cudaMalloc(&gradient_h_output3, memSize);
	cudaMalloc(&gradient_v_output3, memSize);

	cudaMalloc(&d_input4, memSize);
	cudaMalloc(&d_output4, memSize);
	cudaMalloc(&gradient_h_output4, memSize);
	cudaMalloc(&gradient_v_output4, memSize);

		cudaMalloc(&d_input5, memSize);
	cudaMalloc(&d_output5, memSize);
	cudaMalloc(&gradient_h_output5, memSize);
	cudaMalloc(&gradient_v_output5, memSize);

	cudaMalloc(&d_input6, memSize);
	cudaMalloc(&d_output6, memSize);
	cudaMalloc(&gradient_h_output6, memSize);
	cudaMalloc(&gradient_v_output6, memSize);

	cudaMalloc(&d_input7, memSize);
	cudaMalloc(&d_output7, memSize);
	cudaMalloc(&gradient_h_output7, memSize);
	cudaMalloc(&gradient_v_output7, memSize);

	cudaMalloc(&d_input8, memSize);
	cudaMalloc(&d_output8, memSize);
	cudaMalloc(&gradient_h_output8, memSize);
	cudaMalloc(&gradient_v_output8, memSize);

	cudaEventCreate(&start_sobel);
  	cudaEventCreate(&stop_sobel);

	cudaEventRecord(start_sobel, 0);

	cudaMemcpy(d_input, image, memSize, cudaMemcpyHostToDevice);

	cudaMemcpy(d_input2, image2, memSize, cudaMemcpyHostToDevice);

	cudaMemcpy(d_input3, image3, memSize, cudaMemcpyHostToDevice);

	cudaMemcpy(d_input4, image4, memSize, cudaMemcpyHostToDevice);

	cudaMemcpy(d_input5, image5, memSize, cudaMemcpyHostToDevice);

	cudaMemcpy(d_input6, image6, memSize, cudaMemcpyHostToDevice);

	cudaMemcpy(d_input7, image7, memSize, cudaMemcpyHostToDevice);

	cudaMemcpy(d_input8, image8, memSize, cudaMemcpyHostToDevice);
	
	// printf("Launching imageBlur_horizontal \n");
  	imageBlur_horizontal << <blocks, threads >> > (d_input, d_output, WIDTH, HEIGHT);
	  
	imageBlur_vertical << <blocks, threads >> > (d_input, d_output, WIDTH, HEIGHT);

	cudaThreadSynchronize();
	
	imageBlur_horizontal << <blocks, threads >> > (d_input2, d_output2, WIDTH, HEIGHT);

	imageBlur_vertical << <blocks, threads >> > (d_input2, d_output2, WIDTH, HEIGHT);

	cudaThreadSynchronize();

	imageBlur_horizontal << <blocks, threads >> > (d_input3, d_output3, WIDTH, HEIGHT);
	
	imageBlur_vertical << <blocks, threads >> > (d_input3, d_output3, WIDTH, HEIGHT);
	
	cudaThreadSynchronize();

	imageBlur_horizontal << <blocks, threads >> > (d_input4, d_output4, WIDTH, HEIGHT);
	
	imageBlur_vertical << <blocks, threads >> > (d_input4, d_output4, WIDTH, HEIGHT);
	
  	cudaThreadSynchronize();

	  	imageBlur_horizontal << <blocks, threads >> > (d_input5, d_output5, WIDTH, HEIGHT);
	  
	imageBlur_vertical << <blocks, threads >> > (d_input5, d_output5, WIDTH, HEIGHT);

	cudaThreadSynchronize();
	
	imageBlur_horizontal << <blocks, threads >> > (d_input6, d_output6, WIDTH, HEIGHT);

	imageBlur_vertical << <blocks, threads >> > (d_input6, d_output6, WIDTH, HEIGHT);

	cudaThreadSynchronize();

	imageBlur_horizontal << <blocks, threads >> > (d_input7, d_output7, WIDTH, HEIGHT);
	
	imageBlur_vertical << <blocks, threads >> > (d_input7, d_output7, WIDTH, HEIGHT);
	
	cudaThreadSynchronize();

	imageBlur_horizontal << <blocks, threads >> > (d_input8, d_output8, WIDTH, HEIGHT);
	
	imageBlur_vertical << <blocks, threads >> > (d_input8, d_output8, WIDTH, HEIGHT);
	
  	cudaThreadSynchronize();
	
	// printf("Launching gradient_horizontal \n");
	gradient_horizontal<< <blocks, threads>> >(d_input, gradient_h_output, WIDTH, HEIGHT);
	// printf("Launching gradient_vertical \n");
	gradient_vertical<< <blocks, threads>> >(d_input, gradient_v_output, WIDTH, HEIGHT);
	// printf("Launching sobelFilter \n");	
	sobelFilter << <blocks, threads >> > (d_input, d_output, gradient_h_output, gradient_v_output, WIDTH, HEIGHT);

	cudaThreadSynchronize();

	// printf("Launching gradient_horizontal \n");
	gradient_horizontal<< <blocks, threads>> >(d_input2, gradient_h_output2, WIDTH, HEIGHT);
	// printf("Launching gradient_vertical \n");
	gradient_vertical<< <blocks, threads>> >(d_input2, gradient_v_output2, WIDTH, HEIGHT);
	// printf("Launching sobelFilter \n");	
	sobelFilter << <blocks, threads >> > (d_input2, d_output2, gradient_h_output2, gradient_v_output2, WIDTH, HEIGHT);

	cudaThreadSynchronize();

	// printf("Launching gradient_horizontal \n");
	gradient_horizontal<< <blocks, threads>> >(d_input3, gradient_h_output3, WIDTH, HEIGHT);
	// printf("Launching gradient_vertical \n");
	gradient_vertical<< <blocks, threads>> >(d_input3, gradient_v_output3, WIDTH, HEIGHT);
	// printf("Launching sobelFilter \n");	
	sobelFilter << <blocks, threads >> > (d_input3, d_output3, gradient_h_output3, gradient_v_output3, WIDTH, HEIGHT);

	cudaThreadSynchronize();

		// printf("Launching gradient_horizontal \n");
	gradient_horizontal<< <blocks, threads>> >(d_input4, gradient_h_output4, WIDTH, HEIGHT);
	// printf("Launching gradient_vertical \n");
	gradient_vertical<< <blocks, threads>> >(d_input4, gradient_v_output4, WIDTH, HEIGHT);
	// printf("Launching sobelFilter \n");	
	sobelFilter << <blocks, threads >> > (d_input4, d_output4, gradient_h_output4, gradient_v_output4, WIDTH, HEIGHT);

	cudaThreadSynchronize();
	// printf("Launching gradient_horizontal \n");
	gradient_horizontal<< <blocks, threads>> >(d_input, gradient_h_output, WIDTH, HEIGHT);
	// printf("Launching gradient_vertical \n");
	gradient_vertical<< <blocks, threads>> >(d_input, gradient_v_output, WIDTH, HEIGHT);
	// printf("Launching sobelFilter \n");	
	sobelFilter << <blocks, threads >> > (d_input, d_output, gradient_h_output, gradient_v_output, WIDTH, HEIGHT);

	cudaThreadSynchronize();

	// printf("Launching gradient_horizontal \n");
	gradient_horizontal<< <blocks, threads>> >(d_input2, gradient_h_output2, WIDTH, HEIGHT);
	// printf("Launching gradient_vertical \n");
	gradient_vertical<< <blocks, threads>> >(d_input2, gradient_v_output2, WIDTH, HEIGHT);
	// printf("Launching sobelFilter \n");	
	sobelFilter << <blocks, threads >> > (d_input2, d_output2, gradient_h_output2, gradient_v_output2, WIDTH, HEIGHT);

	cudaThreadSynchronize();

	// printf("Launching gradient_horizontal \n");
	gradient_horizontal<< <blocks, threads>> >(d_input3, gradient_h_output3, WIDTH, HEIGHT);
	// printf("Launching gradient_vertical \n");
	gradient_vertical<< <blocks, threads>> >(d_input3, gradient_v_output3, WIDTH, HEIGHT);
	// printf("Launching sobelFilter \n");	
	sobelFilter << <blocks, threads >> > (d_input3, d_output3, gradient_h_output3, gradient_v_output3, WIDTH, HEIGHT);

	cudaThreadSynchronize();

	// printf("Launching gradient_horizontal \n");
	gradient_horizontal<< <blocks, threads>> >(d_input5, gradient_h_output5, WIDTH, HEIGHT);
	// printf("Launching gradient_vertical \n");
	gradient_vertical<< <blocks, threads>> >(d_input5, gradient_v_output5, WIDTH, HEIGHT);
	// printf("Launching sobelFilter \n");	
	sobelFilter << <blocks, threads >> > (d_input5, d_output5, gradient_h_output5, gradient_v_output5, WIDTH, HEIGHT);

	cudaThreadSynchronize();
	
	// printf("Copying data back to host \n");
	cudaMemcpy(final, d_output, memSize, cudaMemcpyDeviceToHost);

	cudaMemcpy(final2, d_output2, memSize, cudaMemcpyDeviceToHost);

	cudaMemcpy(final3, d_output3, memSize, cudaMemcpyDeviceToHost);

	cudaMemcpy(final4, d_output4, memSize, cudaMemcpyDeviceToHost);

	cudaMemcpy(final5, d_output5, memSize, cudaMemcpyDeviceToHost);

	cudaMemcpy(final6, d_output6, memSize, cudaMemcpyDeviceToHost);

	cudaMemcpy(final7, d_output7, memSize, cudaMemcpyDeviceToHost);

	cudaMemcpy(final8, d_output8, memSize, cudaMemcpyDeviceToHost);

	cudaEventRecord(stop_sobel, 0);
  	cudaEventSynchronize(stop_sobel);
  	cudaEventElapsedTime(&sobel, start_sobel, stop_sobel);

	printf("Total Device Time:  %f s \n", sobel/1000);

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

	cudaFree(d_input2);
	cudaFree(d_input2);
	cudaFree(d_output2);
	cudaFree(gradient_h_output2);
	cudaFree(gradient_v_output2);

	cudaFree(d_input3);
	cudaFree(d_input3);
	cudaFree(d_output3);
	cudaFree(gradient_h_output3);
	cudaFree(gradient_v_output3);

	cudaFree(d_input4);
	cudaFree(d_input4);
	cudaFree(d_output4);
	cudaFree(gradient_h_output4);
	cudaFree(gradient_v_output4);

	cudaFree(d_input5);
	cudaFree(d_input5);
	cudaFree(d_output5);
	cudaFree(gradient_h_output5);
	cudaFree(gradient_v_output5);

	cudaFree(d_input6);
	cudaFree(d_input6);
	cudaFree(d_output6);
	cudaFree(gradient_h_output6);
	cudaFree(gradient_v_output6);

	cudaFree(d_input7);
	cudaFree(d_input7);
	cudaFree(d_output7);
	cudaFree(gradient_h_output7);
	cudaFree(gradient_v_output7);

	cudaFree(d_input8);
	cudaFree(d_input8);
	cudaFree(d_output8);
	cudaFree(gradient_h_output8);
	cudaFree(gradient_v_output8);

 
  cudaEventRecord(stop_total, 0);
  cudaEventSynchronize(stop_total);
  cudaEventElapsedTime(&total, start_total, stop_total);

  printf("Total Time:  %f s \n", total/1000);
  
  // write image
  pgmwrite("../images/image-output_g_apollonian_gasket.ascii.pgm", (void *)final,width, height);
  pgmwrite("../images/image-output_g2_apollonian_gasket.ascii.pgm", (void *)final2,width, height);
  pgmwrite("../images/image-output_g3_apollonian_gasket.ascii.pgm", (void *)final3,width, height);
  pgmwrite("../images/image-output_g4_apollonian_gasket.ascii.pgm", (void *)final4,width, height);
  pgmwrite("../images/image-output_g5_apollonian_gasket.ascii.pgm", (void *)final5,width, height);
  pgmwrite("../images/image-output_g6_apollonian_gasket.ascii.pgm", (void *)final6,width, height);
  pgmwrite("../images/image-output_g7_apollonian_gasket.ascii.pgm", (void *)final7,width, height);
  pgmwrite("../images/image-output_g8_apollonian_gasket.ascii.pgm", (void *)final8,width, height);
    
    
	cudaDeviceReset();
	
	return 0;
}

