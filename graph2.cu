
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
#include <helper_cuda.h>
#include <cuda_runtime_api.h>
#include <pgmio.h>
#include <vector>

// Block width WIDTH & HEIGHT
#define BLOCK_W 10
#define BLOCK_H 10

// prototype declarations

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

int main(int argc, char *argv[])
{
	size_t width = 600, height = 600;
	int devID = findCudaDevice(argc, (const char **)argv);
	float *image = NULL, *final = NULL;
	float *image2 = NULL, *final2 = NULL;
	size_t memSize = width * height * sizeof(float);
	checkCudaErrors((cudaMallocHost(&image, memSize)));
	checkCudaErrors((cudaMallocHost(&image2, memSize)));
	checkCudaErrors((cudaMallocHost(&final, memSize)));
	checkCudaErrors((cudaMallocHost(&final2, memSize)));

	// read image 
	pgmread("../images/test_images/apollonian_gasket.ascii.pgm", (void *)image, width, height);
	pgmread("../images/test_images/apollonian_gasket.ascii.pgm", (void *)image2, width, height);

	cudaEventCreate(&start_total);
	cudaEventCreate(&stop_total);
	cudaEventRecord(start_total, 0);

	int x, y;
	float *d_input, *d_output, *gradient_h_output, *gradient_v_output;
	float *d_input2, *d_output2, *gradient_h_output2, *gradient_v_output2;

	printf("Block size: %dx%d\n", BLOCK_W, BLOCK_H);

	checkCudaErrors(cudaMalloc(&d_input, memSize));
	checkCudaErrors(cudaMalloc(&d_output, memSize));
	checkCudaErrors(cudaMalloc(&gradient_h_output, memSize));
	checkCudaErrors(cudaMalloc(&gradient_v_output, memSize));

	checkCudaErrors(cudaMalloc(&d_input2, memSize));
	checkCudaErrors(cudaMalloc(&d_output2, memSize));
	checkCudaErrors(cudaMalloc(&gradient_h_output2, memSize));
	checkCudaErrors(cudaMalloc(&gradient_v_output2, memSize));

	printf("Blocks per grid (width): %d |", (width / BLOCK_W));
	printf("Blocks per grid (height): %d \n", (height / BLOCK_H));

  	cudaGraph_t graph;
  	std::vector<cudaGraphNode_t> nodeDependencies;
	std::vector<cudaGraphNode_t> nodeDependencies2;

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

	cudaMemcpy3DParms memcpyParams01 = {0};

	memcpyParams01.srcArray = NULL;
  	memcpyParams01.srcPos = make_cudaPos(0, 0, 0);
  	memcpyParams01.srcPtr =
      make_cudaPitchedPtr(image2, memSize, 1, 1);
  	memcpyParams01.dstArray = NULL;
  	memcpyParams01.dstPos = make_cudaPos(0, 0, 0);
  	memcpyParams01.dstPtr =
      make_cudaPitchedPtr(d_input2, memSize, 1, 1);
  	memcpyParams01.extent = make_cudaExtent(memSize, 1, 1);
  	memcpyParams01.kind = cudaMemcpyHostToDevice;

	cudaGraphNode_t memcpyNode2;
	checkCudaErrors(
      cudaGraphAddMemcpyNode(&memcpyNode2, graph, NULL, 0, &memcpyParams01));
	nodeDependencies2.push_back(memcpyNode2);

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
    cudaGraphAddKernelNode(&kernelNode, graph, &memcpyNode,
                             1, &kernelNodeParams));

  	nodeDependencies.clear();
  	nodeDependencies.push_back(kernelNode);
  
	cudaGraphNode_t kernelNode2;
	cudaKernelNodeParams kernelNodeParams01 = {0};

	void* kernelArgs1[4] = {(void *)&d_input2,(void *)&d_output2, &width, &height};
	kernelNodeParams01.func = (void *)imageBlur_horizontal;
 	kernelNodeParams01.gridDim = blocks;
  	kernelNodeParams01.blockDim = threads;
  	kernelNodeParams01.sharedMemBytes = 0;
 	kernelNodeParams01.kernelParams = (void **)kernelArgs1;
 	kernelNodeParams01.extra = NULL;

	checkCudaErrors(
    cudaGraphAddKernelNode(&kernelNode2, graph, &memcpyNode2,
                             1, &kernelNodeParams));
  	nodeDependencies2.clear();
  	nodeDependencies2.push_back(kernelNode2);
	
	kernelNodeParams.func = (void *)imageBlur_vertical;

	checkCudaErrors(
    cudaGraphAddKernelNode(&kernelNode, graph, &memcpyNode,
                             1, &kernelNodeParams));

  	nodeDependencies.push_back(kernelNode);

	kernelNodeParams01.func = (void *)imageBlur_vertical;
	
	checkCudaErrors(
    cudaGraphAddKernelNode(&kernelNode2, graph, &memcpyNode2,
                             1, &kernelNodeParams01));

  	nodeDependencies2.push_back(kernelNode2);

	cudaGraphNode_t empty_node;
	checkCudaErrors(
      cudaGraphAddEmptyNode(&empty_node, graph, nodeDependencies.data(),
                             nodeDependencies.size()));

	cudaGraphNode_t empty_node2;
	checkCudaErrors(
      cudaGraphAddEmptyNode(&empty_node2, graph, nodeDependencies2.data(),
                             nodeDependencies2.size()));

	void* kernelArgs2[4] = {(void *)&d_input, (void *)&gradient_h_output, &width, &height};
	kernelNodeParams.func = (void *)gradient_horizontal;
 	kernelNodeParams.kernelParams = (void **)kernelArgs2;
	
	checkCudaErrors(
    cudaGraphAddKernelNode(&kernelNode, graph, &empty_node,
                             1, &kernelNodeParams));

  	nodeDependencies.clear();
  	nodeDependencies.push_back(kernelNode);

	void* kernelArgs21[4] = {(void *)&d_input2, (void *)&gradient_h_output2, &width, &height};
	kernelNodeParams01.func = (void *)gradient_horizontal;
 	kernelNodeParams01.kernelParams = (void **)kernelArgs21;
	
	checkCudaErrors(
    cudaGraphAddKernelNode(&kernelNode2, graph, &empty_node2,
                             1, &kernelNodeParams01));

  	nodeDependencies2.clear();
  	nodeDependencies2.push_back(kernelNode2);
	
	void* kernelArgs3[4] = {(void *)&d_input,(void *)& gradient_v_output, &width, &height};
	kernelNodeParams.func = (void *)gradient_vertical;
 	kernelNodeParams.kernelParams = (void **)kernelArgs3;

	checkCudaErrors(
    cudaGraphAddKernelNode(&kernelNode, graph, &empty_node,
                             1, &kernelNodeParams));

  	nodeDependencies.push_back(kernelNode);

	void* kernelArgs31[4] = {(void *)&d_input2,(void *)& gradient_v_output2, &width, &height};
	kernelNodeParams.func = (void *)gradient_vertical;
 	kernelNodeParams.kernelParams = (void **)kernelArgs31;

	checkCudaErrors(
    cudaGraphAddKernelNode(&kernelNode2, graph, &empty_node2,
                             1, &kernelNodeParams01));

  	nodeDependencies2.push_back(kernelNode2);

	void* kernelArgs4[6] = {(void *)&d_input, (void *)&d_output, (void *)&gradient_h_output, (void *)&gradient_v_output, &width, &height};
	kernelNodeParams.func = (void *)sobelFilter;
 	kernelNodeParams.kernelParams = (void **)kernelArgs4;

	checkCudaErrors(
    cudaGraphAddKernelNode(&kernelNode, graph, nodeDependencies.data(),
                             nodeDependencies.size(), &kernelNodeParams));

  	nodeDependencies.clear();
  	nodeDependencies.push_back(kernelNode);

	void* kernelArgs41[6] = {(void *)&d_input2, (void *)&d_output2, (void *)&gradient_h_output2, (void *)&gradient_v_output2, &width, &height};
	kernelNodeParams01.func = (void *)sobelFilter;
 	kernelNodeParams01.kernelParams = (void **)kernelArgs41;

	checkCudaErrors(
    cudaGraphAddKernelNode(&kernelNode2, graph, nodeDependencies2.data(),
                             nodeDependencies2.size(), &kernelNodeParams01));

  	nodeDependencies2.clear();
  	nodeDependencies2.push_back(kernelNode2);

	memcpyParams.srcPtr = make_cudaPitchedPtr(d_output, memSize, 1, 1);
	memcpyParams.dstPtr = make_cudaPitchedPtr(final, memSize, 1, 1);
	memcpyParams.extent = make_cudaExtent(memSize, 1, 1);
	memcpyParams.kind = cudaMemcpyDeviceToHost;

	checkCudaErrors(
      cudaGraphAddMemcpyNode(&memcpyNode, graph, &kernelNode,
                             1, &memcpyParams));


	memcpyParams01.srcPtr = make_cudaPitchedPtr(d_output2, memSize, 1, 1);
	memcpyParams01.dstPtr = make_cudaPitchedPtr(final2, memSize, 1, 1);
	memcpyParams01.extent = make_cudaExtent(memSize, 1, 1);
	memcpyParams01.kind = cudaMemcpyDeviceToHost;

	checkCudaErrors(
      cudaGraphAddMemcpyNode(&memcpyNode2, graph, &kernelNode2,
                             1, &memcpyParams01));

	checkCudaErrors(cudaGraphDebugDotPrint(graph, "mainGraph.dot", 0));


	cudaGraphExec_t graphExec;
  	checkCudaErrors(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));
	
	warm_up_gpu << <blocks, threads >> > ();

	cudaEventCreate(&start_sobel);
  	cudaEventCreate(&stop_sobel);

    cudaEventRecord(start_sobel, 0);
	
	checkCudaErrors(cudaGraphLaunch(graphExec, 0));
	checkCudaErrors(cudaStreamSynchronize(0));

	cudaEventRecord(stop_sobel, 0);
  	cudaEventSynchronize(stop_sobel);
  	cudaEventElapsedTime(&sobel, start_sobel, stop_sobel);

	printf("Device Time:  %f s \n", sobel/1000);

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

	cudaFree(d_input2);
	cudaFree(d_output2);
	cudaFree(gradient_h_output2);
	cudaFree(gradient_v_output2);


   
  cudaEventRecord(stop_total, 0);
  cudaEventSynchronize(stop_total);
  cudaEventElapsedTime(&total, start_total, stop_total);

  printf("Total Time:  %f s \n", total/1000);
  
	// write image
	pgmwrite("../images/image-output_g_apollonian_gasket.ascii.pgm", (void *)final,width, height);
	pgmwrite("../images/image-output2_g_apollonian_gasket.ascii.pgm", (void *)final2,width, height);
	cudaDeviceReset();
	
	return 0;


}

