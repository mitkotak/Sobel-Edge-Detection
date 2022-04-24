
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifdef __INTELLISENSE__
void __syncthreads();
#endif

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h> 
#include <string.h>
#include <list>  
#include <cuda.h>
#include <helper_cuda.h>
#include <device_functions.h>
#include <helper_cuda.h>
#include <cuda_runtime_api.h>
#include <pgmio.h>
#include <vector>

// Block width WIDTH & HEIGHT
#define BLOCK_W 6
#define BLOCK_H 6

// prototype declarations

void graph_maker(cudaGraph_t graph, float* image, float* final, float* d_input, float* d_output,
float* gradient_h_output, float* gradient_v_output, int width, int height);

#define MAXLINE 128

float total, sobel;
cudaEvent_t start_total, stop_total;
cudaEvent_t start_sobel, stop_sobel;

__global__ void imageBlur_horizontal(float *input, float *output, size_t width, size_t height) {

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

__global__ void imageBlur_vertical(float *input, float *output, size_t width, size_t height) {
	
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

__global__ void gradient_horizontal(float *input, float *output, size_t width, size_t height) {

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


__global__ void gradient_vertical(float *input, float *output, size_t width, size_t height) {

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

__global__ void sobelFilter(float *input, float *output, float *gradient_h_output, float *gradient_v_output, size_t width, size_t height) {
	
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



	//pgmread("../images/apollonian_gasket.ascii.pgm", (void *)image, WIDTH, HEIGHT);
	// pgmread("../images/image20000x20000.pgm", (void *)image, WIDTH, HEIGHT);
	// pgmread("../images/image16384x16384.pgm", (void *)image, WIDTH, HEIGHT);
	// pgmread("image10000x10000.pgm", (void *)image, WIDTH, HEIGHT);
	// pgmread("../images/image4096x4096.pgm", (void *)image, WIDTH, HEIGHT);
	// pgmread("../images/rabbit2000x3000.pgm", (void *)image, WIDTH, HEIGHT);
	// pgmread("../images/image2048x2048.pgm", (void *)image, WIDTH, HEIGHT);
	// pgmread("../images/image1024x1024.pgm", (void *)image, WIDTH, HEIGHT);
	// pgmread("../images/image512x512.pgm", (void *)image, WIDTH, HEIGHT);
	// pgmread("pgmimg.pgm", (void *)image, WIDTH, HEIGHT);

	// pgmwrite("../images/image-output_g_20000x20000.pgm", (void *)final, WIDTH, HEIGHT);
	// pgmwrite("../images/image-output_g_16384x16384.pgm", (void *)final, WIDTH, HEIGHT);
	// pgmwrite("image-outputl10000x1000.pgm", (void *)final, WIDTH, HEIGHT);
	// pgmwrite("../images/image-output_g_4096x4096.pgm", (void *)final, WIDTH, HEIGHT);
	// pgmwrite("../images/image-output_g_2048x2048.pgm", (void *)final, WIDTH, HEIGHT);
	// pgmwrite("../images/rabbit_g_2000x3000.pgm", (void *)final, WIDTH, HEIGHT);
	// pgmwrite("../images/image-output_g_1024x1024.pgm", (void *)final, WIDTH, HEIGHT);
	// pgmwrite("../images/image-output_g_512x512.pgm", (void *)final, WIDTH, HEIGHT);
	// pgmwrite("pgmimg-output.pgm", (void *)final, WIDTH, HEIGHT);

void graph_maker(cudaGraph_t* parent_graph, float* image, float* final, float* d_input, float* d_output,
float* gradient_h_output, float* gradient_v_output, int width, int height){

	cudaGraph_t graph = *parent_graph;
	size_t memSize = width * height * sizeof(float);

  	std::vector<cudaGraphNode_t> nodeDependencies;
	std::vector<cudaGraphNode_t> nodeDependencies2;
	
	// cudaGraphNode_t memcpyNode;
	// cudaMemcpy3DParms memcpyParams = {0};

	// memcpyParams.srcArray = NULL;
  	// memcpyParams.srcPos = make_cudaPos(0, 0, 0);
  	// memcpyParams.srcPtr =
    //   make_cudaPitchedPtr(image, memSize, 1, 1);
  	// memcpyParams.dstArray = NULL;
  	// memcpyParams.dstPos = make_cudaPos(0, 0, 0);
  	// memcpyParams.dstPtr =
    //   make_cudaPitchedPtr(d_input, memSize, 1, 1);
  	// memcpyParams.extent = make_cudaExtent(memSize, 1, 1);
  	// memcpyParams.kind = cudaMemcpyHostToDevice;

	// checkCudaErrors(
    //   cudaGraphAddMemcpyNode(&memcpyNode, graph, NULL, 0, &memcpyParams));
	// nodeDependencies.push_back(memcpyNode);


	dim3 threads(BLOCK_W, BLOCK_H); // threads per block
	dim3 blocks(width / BLOCK_W, height / BLOCK_H); // blocks per grid 

	cudaGraphNode_t kernelNode;
	cudaKernelNodeParams kernelNodeParams = {0};

	void* kernelArgs0[4] = {(void *)&d_input,(void *)&d_output, &width, &height};
	kernelNodeParams.func = (void *)imageBlur_horizontal;
 	kernelNodeParams.gridDim = blocks;
  	kernelNodeParams.blockDim = threads;
  	kernelNodeParams.sharedMemBytes = 0;
 	kernelNodeParams.kernelParams = (void **)kernelArgs0;
 	kernelNodeParams.extra = NULL;


	checkCudaErrors(
    cudaGraphAddKernelNode(&kernelNode, graph, NULL,
                             0, &kernelNodeParams));

  	nodeDependencies2.push_back(kernelNode);
  

	cudaKernelNodeParams kernelNodeParams1 = {0};
	void* kernelArgs1[4] = {(void *)&d_input,(void *)&d_output, &width, &height};
	kernelNodeParams1.func = (void *)imageBlur_vertical;
 	kernelNodeParams1.gridDim = blocks;
  	kernelNodeParams1.blockDim = threads;
  	kernelNodeParams1.sharedMemBytes = 0;
 	kernelNodeParams1.kernelParams = (void **)kernelArgs1;
 	kernelNodeParams1.extra = NULL;

	checkCudaErrors(
    cudaGraphAddKernelNode(&kernelNode, graph, NULL,
                             0, &kernelNodeParams1));

  	nodeDependencies2.push_back(kernelNode);


	cudaKernelNodeParams kernelNodeParams2 = {0};
	void* kernelArgs2[4] = {(void *)&d_input, (void *)&gradient_h_output, &width, &height};
	kernelNodeParams2.func = (void *)gradient_horizontal;
 	kernelNodeParams2.gridDim = blocks;
  	kernelNodeParams2.blockDim = threads;
  	kernelNodeParams2.sharedMemBytes = 0;
 	kernelNodeParams2.kernelParams = (void **)kernelArgs2;
 	kernelNodeParams2.extra = NULL;

	checkCudaErrors(
    cudaGraphAddKernelNode(&kernelNode, graph, nodeDependencies2.data(),
                             nodeDependencies2.size(), &kernelNodeParams2));

  	nodeDependencies.push_back(kernelNode);
	
	cudaKernelNodeParams kernelNodeParams3 = {0};
	void* kernelArgs3[4] = {(void *)&d_input,(void *)& gradient_v_output, &width, &height};
	kernelNodeParams3.func = (void *)gradient_vertical;
 	kernelNodeParams3.gridDim = blocks;
  	kernelNodeParams3.blockDim = threads;
  	kernelNodeParams3.sharedMemBytes = 0;
 	kernelNodeParams3.kernelParams = (void **)kernelArgs3;
 	kernelNodeParams3.extra = NULL;

	checkCudaErrors(
    cudaGraphAddKernelNode(&kernelNode, graph, nodeDependencies2.data(),
                             nodeDependencies2.size(), &kernelNodeParams3));

  	nodeDependencies.push_back(kernelNode);


	cudaKernelNodeParams kernelNodeParams4 = {0};
	void* kernelArgs4[6] = {(void *)&d_input, (void *)&d_output, (void *)&gradient_h_output, (void *)&gradient_v_output, &width, &height};
	kernelNodeParams4.func = (void *)sobelFilter;
 	kernelNodeParams4.gridDim = blocks;
  	kernelNodeParams4.blockDim = threads;
  	kernelNodeParams4.sharedMemBytes = 0;
 	kernelNodeParams4.kernelParams = (void **)kernelArgs4;
 	kernelNodeParams4.extra = NULL;

	checkCudaErrors(
    cudaGraphAddKernelNode(&kernelNode, graph, nodeDependencies.data(),
                             nodeDependencies.size(), &kernelNodeParams4));

  	nodeDependencies.clear();
  	nodeDependencies.push_back(kernelNode);

	// cudaThreadSynchronize();
	// cudaMemcpy3DParms memcpyParams2 = {0};
	// memcpyParams2.srcArray = NULL;
	// memcpyParams2.srcPos = make_cudaPos(0, 0, 0);
	// memcpyParams2.srcPtr = make_cudaPitchedPtr(d_output, memSize, 1, 1);
	// memcpyParams2.dstArray = NULL;
	// memcpyParams2.dstPos = make_cudaPos(0, 0, 0);
	// memcpyParams2.dstPtr = make_cudaPitchedPtr(final, memSize, 1, 1);
	// memcpyParams2.extent = make_cudaExtent(memSize, 1, 1);
	// memcpyParams2.kind = cudaMemcpyDeviceToHost;

	// checkCudaErrors(
    //   cudaGraphAddMemcpyNode(&memcpyNode, graph, &kernelNode,
    //                          1, &memcpyParams2));

	*parent_graph = graph;
}

int main(int argc, char *argv[])
{	FILE *fp;
	fp=fopen("test/benchmark_graph.csv","w+");
	fprintf(fp,"Nimages, Time\n");

	std::list<int> nimage_list = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024 };
	// std::list<int> nimage_list = {4};
	for (int nimage : nimage_list){
		float avg_sobel = 0.0;

		size_t width = 600, height = 600;
		size_t memSize = width * height * sizeof(float);
		float *image_array[nimage];
		float *final_array[nimage];
		float *d_input_array[nimage];
		float *d_output_array[nimage];
		float *gradient_h_output_array[nimage];
		float *gradient_v_output_array[nimage];

		cudaGraph_t graph;

		checkCudaErrors(cudaGraphCreate(&graph, 0));

		for (int i=1; i <= nimage; i++){
		float *image = NULL;
		checkCudaErrors((cudaMallocHost(&image, memSize)));
		pgmread("images/apollonian_gasket.ascii.pgm", (void *)image, width, height);
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
		cudaMemcpy(d_input, image, memSize, cudaMemcpyHostToDevice);
		graph_maker(&graph, image, final, d_input, d_output, gradient_h_output, gradient_v_output, width, height);
	}

	cudaEventCreate(&start_total);
	cudaEventCreate(&stop_total);
	cudaEventRecord(start_total, 0);

	printf("Block size: %dx%d\n", BLOCK_W, BLOCK_H);

	printf("Blocks per grid (width): %d |", (width / BLOCK_W));
	printf("Blocks per grid (height): %d \n", (height / BLOCK_H));

	checkCudaErrors(cudaGraphDebugDotPrint(graph, "mainGraph.dot", 0));

	cudaGraphExec_t graphExec;
  	checkCudaErrors(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));
	
	
	for (int j=1; j <=20; j++){

		cudaEventCreate(&start_sobel);
  		cudaEventCreate(&stop_sobel);

    	cudaEventRecord(start_sobel, 0);

		checkCudaErrors(cudaGraphLaunch(graphExec, 0));
		checkCudaErrors(cudaStreamSynchronize(0));

		cudaEventRecord(stop_sobel, 0);
  		cudaEventSynchronize(stop_sobel);
  		cudaEventElapsedTime(&sobel, start_sobel, stop_sobel);	
		
		avg_sobel += sobel/20;
	}
	

	printf("Total Avg Device Time:  %f s \n", avg_sobel/1000);

	fprintf(fp,"%d,%f\n", nimage, avg_sobel/1000);

	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err)
	{
		fprintf(stderr, "Cuda error: %s: %s.\n", "Main Loop", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	checkCudaErrors(cudaGraphExecDestroy(graphExec));
  	checkCudaErrors(cudaGraphDestroy(graph));

	for (int i=1; i <= nimage; i++){
		float *d_output = d_output_array[i];
		float* final = final_array[i];
		cudaMemcpy(final, d_output, memSize, cudaMemcpyDeviceToHost);
		float *d_input = d_input_array[i];
		cudaFree(d_input);
		cudaFree(d_output);
		float *gradient_h_output = gradient_h_output_array[i];
		cudaFree(gradient_h_output);
		float *gradient_v_output = gradient_v_output_array[i];
		cudaFree(gradient_v_output);

		// write image
		pgmwrite("images/image-output_g_apollonian_gasket.ascii.pgm", (void *)final,width, height);
	}
   
  cudaEventRecord(stop_total, 0);
  cudaEventSynchronize(stop_total);
  cudaEventElapsedTime(&total, start_total, stop_total);

  printf("Total Time:  %f s \n", total/1000);
  

	}
	
    
	cudaDeviceReset();
	
	return 0;

}

