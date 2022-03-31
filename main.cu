
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
#define WIDTH 256
#define HEIGHT 256

// Block width WIDTH & HEIGHT
#define BLOCK_W 16
#define BLOCK_H 16

// buffer to read image into
float image[HEIGHT][WIDTH];

// buffer for resulting image
float final[HEIGHT][WIDTH];

// prototype declarations

void load_image();
void call_kernel();
void save_image();

#define MAXLINE 128

float total, sobel;
cudaEvent_t start_total, stop_total;
cudaEvent_t start_sobel, stop_sobel;


__global__ void imageBlur_horizontal(float *input, float *output, size_t width, size_t height) {

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

__global__ void imageBlur_vertical(float *input, float *output, size_t width, size_t height) {
	
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

__global__ void gradient_horizontal(float *input, float *output, size_t width, size_t height) {

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


__global__ void gradient_vertical(float *input, float *output, size_t width, size_t height) {

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

__global__ void sobelFilter(float *input, float *output, float *gradient_h_output, float *gradient_v_output, size_t width, size_t height) {
	
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


void load_image() {
	pgmread("image512x512.pgm", (void *)image, WIDTH, HEIGHT);
	// pgmread("pgmimg.pgm", (void *)image, WIDTH, HEIGHT);
}

void save_image() {
	pgmwrite("image-output512x512.pgm", (void *)final, WIDTH, HEIGHT);
	// pgmwrite("pgmimg-output.pgm", (void *)final, WIDTH, HEIGHT);
}

void call_kernel() {
	size_t width = WIDTH, height=HEIGHT;
	int x, y;
	float *d_input, *d_output, *gradient_h_output, *gradient_v_output;

	printf("Block size: %dx%d\n", BLOCK_W, BLOCK_H);

	size_t memSize = WIDTH * HEIGHT;

	checkCudaErrors(cudaMalloc(&d_input, memSize));
	checkCudaErrors(cudaMalloc(&d_output, memSize));
	checkCudaErrors(cudaMalloc(&gradient_h_output, memSize));
	checkCudaErrors(cudaMalloc(&gradient_v_output, memSize));

	printf("Blocks per grid (width): %d |", (WIDTH / BLOCK_W));
	printf("Blocks per grid (height): %d |", (HEIGHT / BLOCK_H));

	cudaStream_t streamForGraph;
  	cudaGraph_t graph;
  	std::vector<cudaGraphNode_t> nodeDependencies;
  	
	checkCudaErrors(cudaGraphCreate(&graph, 0));
	  
	cudaGraphNode_t memcpyNode;
	cudaMemcpy3DParms memcpyParams = {0};

	memcpyParams.srcArray = NULL;
  	memcpyParams.srcPos = make_cudaPos(0, 0, 0);
  	memcpyParams.srcPtr =
      make_cudaPitchedPtr(image, memSize, 1, 1);
  	memcpyParams.dstArray = NULL;
  	memcpyParams.dstPos = make_cudaPos(0, 0, 0);
  	memcpyParams.dstPtr =
      make_cudaPitchedPtr(d_input, memSize, 1, 1);
  	memcpyParams.extent = make_cudaExtent(memSize, 1, 1);
  	memcpyParams.kind = cudaMemcpyHostToDevice;

	checkCudaErrors(
      cudaGraphAddMemcpyNode(&memcpyNode, graph, NULL, 0, &memcpyParams));
	nodeDependencies.push_back(memcpyNode);

	// cudaMemcpy(d_input, image, memSize, cudaMemcpyHostToDevice);

	dim3 threads(BLOCK_W, BLOCK_H); // threads per block
	dim3 blocks(WIDTH / BLOCK_W, HEIGHT / BLOCK_H); // blocks per grid 

	cudaGraphNode_t kernelNode;
	cudaKernelNodeParams kernelNodeParams = {0};

	void* kernelArgs0[4] = {(void *)&d_input,(void *)&d_output, &width, &height};
	kernelNodeParams.func = (void *)imageBlur_horizontal;
 	kernelNodeParams.gridDim = threads;
  	kernelNodeParams.blockDim = blocks;
  	kernelNodeParams.sharedMemBytes = 0;
 	kernelNodeParams.kernelParams = (void **)kernelArgs0;
 	kernelNodeParams.extra = NULL;
  
  	//imageBlur << <blocks, threads >> > (d_input, d_output, WIDTH, HEIGHT);

	checkCudaErrors(
    cudaGraphAddKernelNode(&kernelNode, graph, nodeDependencies.data(),
                             nodeDependencies.size(), &kernelNodeParams));

  	nodeDependencies.clear();
  	nodeDependencies.push_back(kernelNode);
  
  	//cudaThreadSynchronize();

	void* kernelArgs1[4] = {(void *)&d_input,(void *)&d_output, &width, &height};
	kernelNodeParams.func = (void *)imageBlur_vertical;
 	kernelNodeParams.gridDim = threads;
  	kernelNodeParams.blockDim = blocks;
  	kernelNodeParams.sharedMemBytes = 0;
 	kernelNodeParams.kernelParams = (void **)kernelArgs1;
 	kernelNodeParams.extra = NULL;

	checkCudaErrors(
    cudaGraphAddKernelNode(&kernelNode, graph, nodeDependencies.data(),
                             nodeDependencies.size(), &kernelNodeParams));

  	nodeDependencies.clear();
  	nodeDependencies.push_back(kernelNode);
	
	memcpyParams.srcArray = NULL;
	memcpyParams.srcPos = make_cudaPos(0, 0, 0);
	memcpyParams.srcPtr = make_cudaPitchedPtr(d_input, memSize, 1, 1);
	memcpyParams.dstArray = NULL;
	memcpyParams.dstPos = make_cudaPos(0, 0, 0);
	memcpyParams.dstPtr = make_cudaPitchedPtr(d_output, memSize, 1, 1);
	memcpyParams.extent = make_cudaExtent(memSize, 1, 1);
	memcpyParams.kind = cudaMemcpyDeviceToHost;

	checkCudaErrors(
      cudaGraphAddMemcpyNode(&memcpyNode, graph, nodeDependencies.data(),
                             nodeDependencies.size(), &memcpyParams));
  	nodeDependencies.clear();
  	nodeDependencies.push_back(memcpyNode);
    
  	//cudaMemcpy(d_input, d_output, memSize, cudaMemcpyDeviceToHost);

	void* kernelArgs2[4] = {(void *)&d_input, (void *)&gradient_h_output, &width, &height};
	kernelNodeParams.func = (void *)gradient_horizontal;
 	kernelNodeParams.gridDim = threads;
  	kernelNodeParams.blockDim = blocks;
  	kernelNodeParams.sharedMemBytes = 0;
 	kernelNodeParams.kernelParams = (void **)kernelArgs2;
 	kernelNodeParams.extra = NULL;

	checkCudaErrors(
    cudaGraphAddKernelNode(&kernelNode, graph, nodeDependencies.data(),
                             nodeDependencies.size(), &kernelNodeParams));

  	nodeDependencies.clear();
  	nodeDependencies.push_back(kernelNode);

	//gradient_horizontal<< <blocks, threads>> >(d_input, gradient_h_output, WIDTH, HEIGHT);

	void* kernelArgs3[4] = {(void *)&d_input,(void *)& gradient_v_output, &width, &height};
	kernelNodeParams.func = (void *)gradient_vertical;
 	kernelNodeParams.gridDim = threads;
  	kernelNodeParams.blockDim = blocks;
  	kernelNodeParams.sharedMemBytes = 0;
 	kernelNodeParams.kernelParams = (void **)kernelArgs3;
 	kernelNodeParams.extra = NULL;

	checkCudaErrors(
    cudaGraphAddKernelNode(&kernelNode, graph, nodeDependencies.data(),
                             nodeDependencies.size(), &kernelNodeParams));

  	nodeDependencies.clear();
  	nodeDependencies.push_back(kernelNode);

	//gradient_vertical<< <blocks, threads>> >(d_input, gradient_v_output, WIDTH, HEIGHT);

	void* kernelArgs4[6] = {(void *)&d_input, (void *)&d_output, (void *)&gradient_h_output, (void *)&gradient_v_output, &width, &height};
	kernelNodeParams.func = (void *)sobelFilter;
 	kernelNodeParams.gridDim = threads;
  	kernelNodeParams.blockDim = blocks;
  	kernelNodeParams.sharedMemBytes = 0;
 	kernelNodeParams.kernelParams = (void **)kernelArgs4;
 	kernelNodeParams.extra = NULL;

	checkCudaErrors(
    cudaGraphAddKernelNode(&kernelNode, graph, nodeDependencies.data(),
                             nodeDependencies.size(), &kernelNodeParams));

  	nodeDependencies.clear();
  	nodeDependencies.push_back(kernelNode);

	//sobelFilter << <blocks, threads >> > (d_input, d_output, gradient_h_output, gradient_v_output, WIDTH, HEIGHT);

	//cudaThreadSynchronize();

	memcpyParams.srcArray = NULL;
	memcpyParams.srcPos = make_cudaPos(0, 0, 0);
	memcpyParams.srcPtr = make_cudaPitchedPtr(d_output, memSize, 1, 1);
	memcpyParams.dstArray = NULL;
	memcpyParams.dstPos = make_cudaPos(0, 0, 0);
	memcpyParams.dstPtr = make_cudaPitchedPtr(&final, memSize, 1, 1);
	memcpyParams.extent = make_cudaExtent(memSize, 1, 1);
	memcpyParams.kind = cudaMemcpyDeviceToHost;

	checkCudaErrors(
      cudaGraphAddMemcpyNode(&memcpyNode, graph, nodeDependencies.data(),
                             nodeDependencies.size(), &memcpyParams));
  	nodeDependencies.clear();
  	nodeDependencies.push_back(memcpyNode);

	cudaGraphExec_t graphExec;
  	checkCudaErrors(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));
	checkCudaErrors(cudaStreamSynchronize(0));

	// cudaMemcpy(final, d_output, memSize, cudaMemcpyDeviceToHost);

	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err)
	{
		fprintf(stderr, "Cuda error: %s: %s.\n", "Main Loop", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	checkCudaErrors(cudaGraphExecDestroy(graphExec));
  	checkCudaErrors(cudaGraphDestroy(graph));
	cudaFree(d_input);
	cudaFree(d_output);
	cudaFree(gradient_h_output);
	cudaFree(gradient_v_output);
}

int main(int argc, char *argv[])
{
  cudaEventCreate(&start_total);
  cudaEventCreate(&stop_total);
    
  cudaEventCreate(&start_sobel);
  cudaEventCreate(&stop_sobel);
    
  cudaEventRecord(start_total, 0);

	load_image();
   
  cudaEventRecord(start_sobel, 0);

	call_kernel();
  
  cudaEventRecord(stop_sobel, 0);
  cudaEventSynchronize(stop_sobel);
  cudaEventElapsedTime(&sobel, start_sobel, stop_sobel);

	save_image();
   
  cudaEventRecord(stop_total, 0);
  cudaEventSynchronize(stop_total);
  cudaEventElapsedTime(&total, start_total, stop_total);
    
  printf("Total Parallel Time:  %f s |", sobel/1000);
  printf("Total Serial Time:  %f s |", (total-sobel)/1000);
  printf("Total Time:  %f s |", total/1000);
  
    
	cudaDeviceReset();
	
	return 0;
}

