
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
	float *image3 = NULL, *final3 = NULL;
	float *image4 = NULL, *final4 = NULL;
	float *image5 = NULL, *final5 = NULL;
	float *image6 = NULL, *final6 = NULL;
	float *image7 = NULL, *final7 = NULL;
	float *image8 = NULL, *final8 = NULL;

	size_t memSize = width * height * sizeof(float);
	checkCudaErrors((cudaMallocHost(&image, memSize)));
	checkCudaErrors((cudaMallocHost(&image2, memSize)));
	checkCudaErrors((cudaMallocHost(&final, memSize)));
	checkCudaErrors((cudaMallocHost(&final2, memSize)));
	checkCudaErrors((cudaMallocHost(&image3, memSize)));
	checkCudaErrors((cudaMallocHost(&image4, memSize)));
	checkCudaErrors((cudaMallocHost(&final3, memSize)));
	checkCudaErrors((cudaMallocHost(&final4, memSize)));
	checkCudaErrors((cudaMallocHost(&image5, memSize)));
	checkCudaErrors((cudaMallocHost(&image6, memSize)));
	checkCudaErrors((cudaMallocHost(&final5, memSize)));
	checkCudaErrors((cudaMallocHost(&final6, memSize)));
	checkCudaErrors((cudaMallocHost(&image7, memSize)));
	checkCudaErrors((cudaMallocHost(&image8, memSize)));
	checkCudaErrors((cudaMallocHost(&final7, memSize)));
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

	checkCudaErrors(cudaMalloc(&d_input, memSize));
	checkCudaErrors(cudaMalloc(&d_output, memSize));
	checkCudaErrors(cudaMalloc(&gradient_h_output, memSize));
	checkCudaErrors(cudaMalloc(&gradient_v_output, memSize));

	checkCudaErrors(cudaMalloc(&d_input2, memSize));
	checkCudaErrors(cudaMalloc(&d_output2, memSize));
	checkCudaErrors(cudaMalloc(&gradient_h_output2, memSize));
	checkCudaErrors(cudaMalloc(&gradient_v_output2, memSize));

	checkCudaErrors(cudaMalloc(&d_input3, memSize));
	checkCudaErrors(cudaMalloc(&d_output3, memSize));
	checkCudaErrors(cudaMalloc(&gradient_h_output3, memSize));
	checkCudaErrors(cudaMalloc(&gradient_v_output3, memSize));

	checkCudaErrors(cudaMalloc(&d_input4, memSize));
	checkCudaErrors(cudaMalloc(&d_output4, memSize));
	checkCudaErrors(cudaMalloc(&gradient_h_output4, memSize));
	checkCudaErrors(cudaMalloc(&gradient_v_output4, memSize));

	checkCudaErrors(cudaMalloc(&d_input5, memSize));
	checkCudaErrors(cudaMalloc(&d_output5, memSize));
	checkCudaErrors(cudaMalloc(&gradient_h_output5, memSize));
	checkCudaErrors(cudaMalloc(&gradient_v_output5, memSize));

	checkCudaErrors(cudaMalloc(&d_input6, memSize));
	checkCudaErrors(cudaMalloc(&d_output6, memSize));
	checkCudaErrors(cudaMalloc(&gradient_h_output6, memSize));
	checkCudaErrors(cudaMalloc(&gradient_v_output6, memSize));

	checkCudaErrors(cudaMalloc(&d_input7, memSize));
	checkCudaErrors(cudaMalloc(&d_output7, memSize));
	checkCudaErrors(cudaMalloc(&gradient_h_output7, memSize));
	checkCudaErrors(cudaMalloc(&gradient_v_output7, memSize));

	checkCudaErrors(cudaMalloc(&d_input8, memSize));
	checkCudaErrors(cudaMalloc(&d_output8, memSize));
	checkCudaErrors(cudaMalloc(&gradient_h_output8, memSize));
	checkCudaErrors(cudaMalloc(&gradient_v_output8, memSize));


	printf("Blocks per grid (width): %d |", (width / BLOCK_W));
	printf("Blocks per grid (height): %d \n", (height / BLOCK_H));

  	cudaGraph_t graph;
  	std::vector<cudaGraphNode_t> nodeDependencies;
	std::vector<cudaGraphNode_t> nodeDependencies2;
	std::vector<cudaGraphNode_t> nodeDependencies3;
	std::vector<cudaGraphNode_t> nodeDependencies4;
	std::vector<cudaGraphNode_t> nodeDependencies5;
	std::vector<cudaGraphNode_t> nodeDependencies6;
	std::vector<cudaGraphNode_t> nodeDependencies7;
	std::vector<cudaGraphNode_t> nodeDependencies8;

	checkCudaErrors(cudaGraphCreate(&graph, 0));

	// Copy data from host to device
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

	
	cudaMemcpy3DParms memcpyParams02 = {0};

	memcpyParams02.srcArray = NULL;
  	memcpyParams02.srcPos = make_cudaPos(0, 0, 0);
  	memcpyParams02.srcPtr =
      make_cudaPitchedPtr(image3, memSize, 1, 1);
  	memcpyParams02.dstArray = NULL;
  	memcpyParams02.dstPos = make_cudaPos(0, 0, 0);
  	memcpyParams02.dstPtr =
      make_cudaPitchedPtr(d_input3, memSize, 1, 1);
  	memcpyParams02.extent = make_cudaExtent(memSize, 1, 1);
  	memcpyParams02.kind = cudaMemcpyHostToDevice;

	cudaGraphNode_t memcpyNode3;
	checkCudaErrors(
      cudaGraphAddMemcpyNode(&memcpyNode3, graph, NULL, 0, &memcpyParams02));
	nodeDependencies3.push_back(memcpyNode3);

	cudaMemcpy3DParms memcpyParams03 = {0};
	memcpyParams03.srcArray = NULL;
  	memcpyParams03.srcPos = make_cudaPos(0, 0, 0);
  	memcpyParams03.srcPtr =
      make_cudaPitchedPtr(image4, memSize, 1, 1);
  	memcpyParams03.dstArray = NULL;
  	memcpyParams03.dstPos = make_cudaPos(0, 0, 0);
  	memcpyParams03.dstPtr =
      make_cudaPitchedPtr(d_input4, memSize, 1, 1);
  	memcpyParams03.extent = make_cudaExtent(memSize, 1, 1);
  	memcpyParams03.kind = cudaMemcpyHostToDevice;

	cudaGraphNode_t memcpyNode4;
	checkCudaErrors(
      cudaGraphAddMemcpyNode(&memcpyNode4, graph, NULL, 0, &memcpyParams03));
	nodeDependencies4.push_back(memcpyNode4);

	cudaGraphNode_t memcpyNode5;
	cudaMemcpy3DParms memcpyParams04 = {0};

	memcpyParams04.srcArray = NULL;
  	memcpyParams04.srcPos = make_cudaPos(0, 0, 0);
  	memcpyParams04.srcPtr =
      make_cudaPitchedPtr(image5, memSize, 1, 1);
  	memcpyParams04.dstArray = NULL;
  	memcpyParams04.dstPos = make_cudaPos(0, 0, 0);
  	memcpyParams04.dstPtr =
      make_cudaPitchedPtr(d_input5, memSize, 1, 1);
  	memcpyParams04.extent = make_cudaExtent(memSize, 1, 1);
  	memcpyParams04.kind = cudaMemcpyHostToDevice;

	checkCudaErrors(
      cudaGraphAddMemcpyNode(&memcpyNode5, graph, NULL, 0, &memcpyParams04));
	nodeDependencies5.push_back(memcpyNode5);

	cudaMemcpy3DParms memcpyParams05 = {0};

	memcpyParams05.srcArray = NULL;
  	memcpyParams05.srcPos = make_cudaPos(0, 0, 0);
  	memcpyParams05.srcPtr =
      make_cudaPitchedPtr(image6, memSize, 1, 1);
  	memcpyParams05.dstArray = NULL;
  	memcpyParams05.dstPos = make_cudaPos(0, 0, 0);
  	memcpyParams05.dstPtr =
      make_cudaPitchedPtr(d_input6, memSize, 1, 1);
  	memcpyParams05.extent = make_cudaExtent(memSize, 1, 1);
  	memcpyParams05.kind = cudaMemcpyHostToDevice;

	cudaGraphNode_t memcpyNode6;
	checkCudaErrors(
      cudaGraphAddMemcpyNode(&memcpyNode6, graph, NULL, 0, &memcpyParams05));
	nodeDependencies6.push_back(memcpyNode6);

	
	cudaMemcpy3DParms memcpyParams06 = {0};

	memcpyParams06.srcArray = NULL;
  	memcpyParams06.srcPos = make_cudaPos(0, 0, 0);
  	memcpyParams06.srcPtr =
      make_cudaPitchedPtr(image7, memSize, 1, 1);
  	memcpyParams06.dstArray = NULL;
  	memcpyParams06.dstPos = make_cudaPos(0, 0, 0);
  	memcpyParams06.dstPtr =
      make_cudaPitchedPtr(d_input7, memSize, 1, 1);
  	memcpyParams06.extent = make_cudaExtent(memSize, 1, 1);
  	memcpyParams06.kind = cudaMemcpyHostToDevice;

	cudaGraphNode_t memcpyNode7;
	checkCudaErrors(
      cudaGraphAddMemcpyNode(&memcpyNode7, graph, NULL, 0, &memcpyParams06));
	nodeDependencies7.push_back(memcpyNode7);

	cudaMemcpy3DParms memcpyParams07 = {0};
	memcpyParams07.srcArray = NULL;
  	memcpyParams07.srcPos = make_cudaPos(0, 0, 0);
  	memcpyParams07.srcPtr =
      make_cudaPitchedPtr(image8, memSize, 1, 1);
  	memcpyParams07.dstArray = NULL;
  	memcpyParams07.dstPos = make_cudaPos(0, 0, 0);
  	memcpyParams07.dstPtr =
      make_cudaPitchedPtr(d_input8, memSize, 1, 1);
  	memcpyParams07.extent = make_cudaExtent(memSize, 1, 1);
  	memcpyParams07.kind = cudaMemcpyHostToDevice;

	cudaGraphNode_t memcpyNode8;
	checkCudaErrors(
      cudaGraphAddMemcpyNode(&memcpyNode8, graph, NULL, 0, &memcpyParams07));
	nodeDependencies8.push_back(memcpyNode8);

	dim3 threads(BLOCK_W, BLOCK_H); // threads per block
	dim3 blocks(width / BLOCK_W, height / BLOCK_H); // blocks per grid 

	// imageBlur_horizontal kernel
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

	void* kernelArgs01[4] = {(void *)&d_input2,(void *)&d_output2, &width, &height};
	kernelNodeParams01.func = (void *)imageBlur_horizontal;
 	kernelNodeParams01.gridDim = blocks;
  	kernelNodeParams01.blockDim = threads;
  	kernelNodeParams01.sharedMemBytes = 0;
 	kernelNodeParams01.kernelParams = (void **)kernelArgs01;
 	kernelNodeParams01.extra = NULL;

	checkCudaErrors(
    cudaGraphAddKernelNode(&kernelNode2, graph, &memcpyNode2,
                             1, &kernelNodeParams01));
  	nodeDependencies2.clear();
  	nodeDependencies2.push_back(kernelNode2);

	cudaGraphNode_t kernelNode3;
	cudaKernelNodeParams kernelNodeParams02 = {0};

	void* kernelArgs02[4] = {(void *)&d_input3,(void *)&d_output3, &width, &height};
	kernelNodeParams02.func = (void *)imageBlur_horizontal;
 	kernelNodeParams02.gridDim = blocks;
  	kernelNodeParams02.blockDim = threads;
  	kernelNodeParams02.sharedMemBytes = 0;
 	kernelNodeParams02.kernelParams = (void **)kernelArgs02;
 	kernelNodeParams02.extra = NULL;

	checkCudaErrors(
    cudaGraphAddKernelNode(&kernelNode3, graph, &memcpyNode3,
                             1, &kernelNodeParams02));
  	nodeDependencies3.clear();
  	nodeDependencies3.push_back(kernelNode3);
	
	cudaGraphNode_t kernelNode4;
	cudaKernelNodeParams kernelNodeParams03 = {0};

	void* kernelArgs03[4] = {(void *)&d_input4,(void *)&d_output4, &width, &height};
	kernelNodeParams03.func = (void *)imageBlur_horizontal;
 	kernelNodeParams03.gridDim = blocks;
  	kernelNodeParams03.blockDim = threads;
  	kernelNodeParams03.sharedMemBytes = 0;
 	kernelNodeParams03.kernelParams = (void **)kernelArgs03;
 	kernelNodeParams03.extra = NULL;

	checkCudaErrors(
    cudaGraphAddKernelNode(&kernelNode4, graph, &memcpyNode4,
                             1, &kernelNodeParams03));
  	nodeDependencies4.clear();
  	nodeDependencies4.push_back(kernelNode4);

	cudaGraphNode_t kernelNode5;
	cudaKernelNodeParams kernelNodeParams04 = {0};

	void* kernelArgs04[4] = {(void *)&d_input5,(void *)&d_output5, &width, &height};
	kernelNodeParams04.func = (void *)imageBlur_horizontal;
 	kernelNodeParams04.gridDim = blocks;
  	kernelNodeParams04.blockDim = threads;
  	kernelNodeParams04.sharedMemBytes = 0;
 	kernelNodeParams04.kernelParams = (void **)kernelArgs04;
 	kernelNodeParams04.extra = NULL;


	checkCudaErrors(
    cudaGraphAddKernelNode(&kernelNode5, graph, &memcpyNode,
                             1, &kernelNodeParams04));

  	nodeDependencies5.clear();
  	nodeDependencies5.push_back(kernelNode5);
  
	cudaGraphNode_t kernelNode6;
	cudaKernelNodeParams kernelNodeParams05 = {0};

	void* kernelArgs05[4] = {(void *)&d_input5,(void *)&d_output5, &width, &height};
	kernelNodeParams05.func = (void *)imageBlur_horizontal;
 	kernelNodeParams05.gridDim = blocks;
  	kernelNodeParams05.blockDim = threads;
  	kernelNodeParams05.sharedMemBytes = 0;
 	kernelNodeParams05.kernelParams = (void **)kernelArgs05;
 	kernelNodeParams05.extra = NULL;

	checkCudaErrors(
    cudaGraphAddKernelNode(&kernelNode6, graph, &memcpyNode6,
                             1, &kernelNodeParams05));
  	nodeDependencies6.clear();
  	nodeDependencies6.push_back(kernelNode6);

	cudaGraphNode_t kernelNode7;
	cudaKernelNodeParams kernelNodeParams06 = {0};

	void* kernelArgs06[4] = {(void *)&d_input7,(void *)&d_output7, &width, &height};
	kernelNodeParams06.func = (void *)imageBlur_horizontal;
 	kernelNodeParams06.gridDim = blocks;
  	kernelNodeParams06.blockDim = threads;
  	kernelNodeParams06.sharedMemBytes = 0;
 	kernelNodeParams06.kernelParams = (void **)kernelArgs06;
 	kernelNodeParams06.extra = NULL;

	checkCudaErrors(
    cudaGraphAddKernelNode(&kernelNode7, graph, &memcpyNode7,
                             1, &kernelNodeParams06));
  	nodeDependencies7.clear();
  	nodeDependencies7.push_back(kernelNode7);
	
	cudaGraphNode_t kernelNode8;
	cudaKernelNodeParams kernelNodeParams07 = {0};

	void* kernelArgs07[4] = {(void *)&d_input8,(void *)&d_output8, &width, &height};
	kernelNodeParams07.func = (void *)imageBlur_horizontal;
 	kernelNodeParams07.gridDim = blocks;
  	kernelNodeParams07.blockDim = threads;
  	kernelNodeParams07.sharedMemBytes = 0;
 	kernelNodeParams07.kernelParams = (void **)kernelArgs07;
 	kernelNodeParams07.extra = NULL;

	checkCudaErrors(
    cudaGraphAddKernelNode(&kernelNode8, graph, &memcpyNode8,
                             1, &kernelNodeParams07));
  	nodeDependencies8.clear();
  	nodeDependencies8.push_back(kernelNode8);
	

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

	
	kernelNodeParams02.func = (void *)imageBlur_vertical;
	
	checkCudaErrors(
    cudaGraphAddKernelNode(&kernelNode3, graph, &memcpyNode3,
                             1, &kernelNodeParams02));

  	nodeDependencies3.push_back(kernelNode3);

	
	kernelNodeParams03.func = (void *)imageBlur_vertical;
	
	checkCudaErrors(
    cudaGraphAddKernelNode(&kernelNode4, graph, &memcpyNode4,
                             1, &kernelNodeParams03));

  	nodeDependencies4.push_back(kernelNode4);

	kernelNodeParams04.func = (void *)imageBlur_vertical;

	checkCudaErrors(
    cudaGraphAddKernelNode(&kernelNode5, graph, &memcpyNode5,
                             1, &kernelNodeParams04));

  	nodeDependencies.push_back(kernelNode5);

	kernelNodeParams05.func = (void *)imageBlur_vertical;
	
	checkCudaErrors(
    cudaGraphAddKernelNode(&kernelNode6, graph, &memcpyNode6,
                             1, &kernelNodeParams05));

  	nodeDependencies6.push_back(kernelNode6);

	
	kernelNodeParams06.func = (void *)imageBlur_vertical;
	
	checkCudaErrors(
    cudaGraphAddKernelNode(&kernelNode7, graph, &memcpyNode7,
                             1, &kernelNodeParams06));

  	nodeDependencies7.push_back(kernelNode7);

	
	kernelNodeParams07.func = (void *)imageBlur_vertical;
	
	checkCudaErrors(
    cudaGraphAddKernelNode(&kernelNode8, graph, &memcpyNode8,
                             1, &kernelNodeParams07));

  	nodeDependencies8.push_back(kernelNode8);

	cudaGraphNode_t empty_node;
	checkCudaErrors(
      cudaGraphAddEmptyNode(&empty_node, graph, nodeDependencies.data(),
                             nodeDependencies.size()));

	cudaGraphNode_t empty_node2;
	checkCudaErrors(
      cudaGraphAddEmptyNode(&empty_node2, graph, nodeDependencies2.data(),
                             nodeDependencies2.size()));

	cudaGraphNode_t empty_node3;
	checkCudaErrors(
      cudaGraphAddEmptyNode(&empty_node3, graph, nodeDependencies3.data(),
                             nodeDependencies3.size()));

	cudaGraphNode_t empty_node4;
	checkCudaErrors(
      cudaGraphAddEmptyNode(&empty_node4, graph, nodeDependencies4.data(),
                             nodeDependencies4.size()));

	cudaGraphNode_t empty_node5;
	checkCudaErrors(
      cudaGraphAddEmptyNode(&empty_node5, graph, nodeDependencies5.data(),
                             nodeDependencies5.size()));

	cudaGraphNode_t empty_node6;
	checkCudaErrors(
      cudaGraphAddEmptyNode(&empty_node6, graph, nodeDependencies6.data(),
                             nodeDependencies6.size()));

	cudaGraphNode_t empty_node7;
	checkCudaErrors(
      cudaGraphAddEmptyNode(&empty_node7, graph, nodeDependencies7.data(),
                             nodeDependencies7.size()));

	cudaGraphNode_t empty_node8;
	checkCudaErrors(
      cudaGraphAddEmptyNode(&empty_node8, graph, nodeDependencies8.data(),
                             nodeDependencies8.size()));

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


	void* kernelArgs22[4] = {(void *)&d_input3, (void *)&gradient_h_output3, &width, &height};
	kernelNodeParams02.func = (void *)gradient_horizontal;
 	kernelNodeParams02.kernelParams = (void **)kernelArgs22;
	
	checkCudaErrors(
    cudaGraphAddKernelNode(&kernelNode3, graph, &empty_node3,
                             1, &kernelNodeParams02));

  	nodeDependencies3.clear();
  	nodeDependencies3.push_back(kernelNode3);

	
	void* kernelArgs23[4] = {(void *)&d_input4, (void *)&gradient_h_output4, &width, &height};
	kernelNodeParams03.func = (void *)gradient_horizontal;
 	kernelNodeParams03.kernelParams = (void **)kernelArgs23;
	
	checkCudaErrors(
    cudaGraphAddKernelNode(&kernelNode4, graph, &empty_node4,
                             1, &kernelNodeParams03));

  	nodeDependencies4.clear();
  	nodeDependencies4.push_back(kernelNode4);
	
	void* kernelArgs24[4] = {(void *)&d_input5, (void *)&gradient_h_output5, &width, &height};
	kernelNodeParams04.func = (void *)gradient_horizontal;
 	kernelNodeParams04.kernelParams = (void **)kernelArgs24;
	
	checkCudaErrors(
    cudaGraphAddKernelNode(&kernelNode5, graph, &empty_node5,
                             1, &kernelNodeParams04));

  	nodeDependencies5.clear();
  	nodeDependencies5.push_back(kernelNode5);

	void* kernelArgs25[4] = {(void *)&d_input5, (void *)&gradient_h_output5, &width, &height};
	kernelNodeParams05.func = (void *)gradient_horizontal;
 	kernelNodeParams05.kernelParams = (void **)kernelArgs25;
	
	checkCudaErrors(
    cudaGraphAddKernelNode(&kernelNode6, graph, &empty_node6,
                             1, &kernelNodeParams05));

  	nodeDependencies6.clear();
  	nodeDependencies6.push_back(kernelNode6);


	void* kernelArgs26[4] = {(void *)&d_input7, (void *)&gradient_h_output7, &width, &height};
	kernelNodeParams06.func = (void *)gradient_horizontal;
 	kernelNodeParams06.kernelParams = (void **)kernelArgs26;
	
	checkCudaErrors(
    cudaGraphAddKernelNode(&kernelNode7, graph, &empty_node7,
                             1, &kernelNodeParams06));

  	nodeDependencies7.clear();
  	nodeDependencies7.push_back(kernelNode7);

	
	void* kernelArgs27[4] = {(void *)&d_input8, (void *)&gradient_h_output8, &width, &height};
	kernelNodeParams07.func = (void *)gradient_horizontal;
 	kernelNodeParams07.kernelParams = (void **)kernelArgs27;
	
	checkCudaErrors(
    cudaGraphAddKernelNode(&kernelNode8, graph, &empty_node8,
                             1, &kernelNodeParams07));

  	nodeDependencies4.clear();
  	nodeDependencies4.push_back(kernelNode8);
	
	
	void* kernelArgs3[4] = {(void *)&d_input,(void *)& gradient_v_output, &width, &height};
	kernelNodeParams.func = (void *)gradient_vertical;
 	kernelNodeParams.kernelParams = (void **)kernelArgs3;

	checkCudaErrors(
    cudaGraphAddKernelNode(&kernelNode, graph, &empty_node,
                             1, &kernelNodeParams));

  	nodeDependencies.push_back(kernelNode);

	void* kernelArgs31[4] = {(void *)&d_input2,(void *)& gradient_v_output2, &width, &height};
	kernelNodeParams01.func = (void *)gradient_vertical;
 	kernelNodeParams01.kernelParams = (void **)kernelArgs31;

	checkCudaErrors(
    cudaGraphAddKernelNode(&kernelNode2, graph, &empty_node2,
                             1, &kernelNodeParams01));

  	nodeDependencies2.push_back(kernelNode2);

	
	void* kernelArgs32[4] = {(void *)&d_input3,(void *)& gradient_v_output3, &width, &height};
	kernelNodeParams02.func = (void *)gradient_vertical;
 	kernelNodeParams02.kernelParams = (void **)kernelArgs32;

	checkCudaErrors(
    cudaGraphAddKernelNode(&kernelNode3, graph, &empty_node3,
                             1, &kernelNodeParams02));

  	nodeDependencies3.push_back(kernelNode3);


	void* kernelArgs33[4] = {(void *)&d_input4,(void *)& gradient_v_output4, &width, &height};
	kernelNodeParams03.func = (void *)gradient_vertical;
 	kernelNodeParams03.kernelParams = (void **)kernelArgs33;

	checkCudaErrors(
    cudaGraphAddKernelNode(&kernelNode4, graph, &empty_node4,
                             1, &kernelNodeParams03));

  	nodeDependencies4.push_back(kernelNode4);

	void* kernelArgs34[4] = {(void *)&d_input5,(void *)& gradient_v_output5, &width, &height};
	kernelNodeParams04.func = (void *)gradient_vertical;
 	kernelNodeParams04.kernelParams = (void **)kernelArgs04;

	checkCudaErrors(
    cudaGraphAddKernelNode(&kernelNode5, graph, &empty_node5,
                             1, &kernelNodeParams04));

  	nodeDependencies.push_back(kernelNode5);

	void* kernelArgs35[4] = {(void *)&d_input6,(void *)& gradient_v_output6, &width, &height};
	kernelNodeParams05.func = (void *)gradient_vertical;
 	kernelNodeParams05.kernelParams = (void **)kernelArgs35;

	checkCudaErrors(
    cudaGraphAddKernelNode(&kernelNode6, graph, &empty_node6,
                             1, &kernelNodeParams05));

  	nodeDependencies6.push_back(kernelNode6);

	
	void* kernelArgs36[4] = {(void *)&d_input7,(void *)& gradient_v_output7, &width, &height};
	kernelNodeParams06.func = (void *)gradient_vertical;
 	kernelNodeParams06.kernelParams = (void **)kernelArgs36;

	checkCudaErrors(
    cudaGraphAddKernelNode(&kernelNode7, graph, &empty_node7,
                             1, &kernelNodeParams06));

  	nodeDependencies7.push_back(kernelNode7);


	void* kernelArgs37[4] = {(void *)&d_input8,(void *)& gradient_v_output8, &width, &height};
	kernelNodeParams07.func = (void *)gradient_vertical;
 	kernelNodeParams07.kernelParams = (void **)kernelArgs37;

	checkCudaErrors(
    cudaGraphAddKernelNode(&kernelNode8, graph, &empty_node8,
                             1, &kernelNodeParams07));

  	nodeDependencies8.push_back(kernelNode8);


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

	void* kernelArgs42[6] = {(void *)&d_input3, (void *)&d_output3, (void *)&gradient_h_output3, (void *)&gradient_v_output3, &width, &height};
	kernelNodeParams02.func = (void *)sobelFilter;
 	kernelNodeParams02.kernelParams = (void **)kernelArgs42;

	checkCudaErrors(
    cudaGraphAddKernelNode(&kernelNode3, graph, nodeDependencies3.data(),
                             nodeDependencies3.size(), &kernelNodeParams02));

  	nodeDependencies3.clear();
  	nodeDependencies3.push_back(kernelNode3);


	void* kernelArgs43[6] = {(void *)&d_input4, (void *)&d_output4, (void *)&gradient_h_output4, (void *)&gradient_v_output4, &width, &height};
	kernelNodeParams03.func = (void *)sobelFilter;
 	kernelNodeParams03.kernelParams = (void **)kernelArgs43;

	checkCudaErrors(
    cudaGraphAddKernelNode(&kernelNode4, graph, nodeDependencies4.data(),
                             nodeDependencies4.size(), &kernelNodeParams03));

  	nodeDependencies4.clear();
  	nodeDependencies4.push_back(kernelNode4);
	
	void* kernelArgs44[6] = {(void *)&d_input5, (void *)&d_output5, (void *)&gradient_h_output5, (void *)&gradient_v_output5, &width, &height};
	kernelNodeParams04.func = (void *)sobelFilter;
 	kernelNodeParams04.kernelParams = (void **)kernelArgs44;

	checkCudaErrors(
    cudaGraphAddKernelNode(&kernelNode5, graph, nodeDependencies5.data(),
                             nodeDependencies5.size(), &kernelNodeParams04));

  	nodeDependencies5.clear();
  	nodeDependencies5.push_back(kernelNode5);

	void* kernelArgs45[6] = {(void *)&d_input6, (void *)&d_output6, (void *)&gradient_h_output6, (void *)&gradient_v_output6, &width, &height};
	kernelNodeParams05.func = (void *)sobelFilter;
 	kernelNodeParams05.kernelParams = (void **)kernelArgs45;

	checkCudaErrors(
    cudaGraphAddKernelNode(&kernelNode6, graph, nodeDependencies6.data(),
                             nodeDependencies6.size(), &kernelNodeParams05));

  	nodeDependencies6.clear();
  	nodeDependencies6.push_back(kernelNode6);

	void* kernelArgs46[6] = {(void *)&d_input7, (void *)&d_output7, (void *)&gradient_h_output7, (void *)&gradient_v_output7, &width, &height};
	kernelNodeParams06.func = (void *)sobelFilter;
 	kernelNodeParams06.kernelParams = (void **)kernelArgs46;

	checkCudaErrors(
    cudaGraphAddKernelNode(&kernelNode7, graph, nodeDependencies7.data(),
                             nodeDependencies7.size(), &kernelNodeParams06));

  	nodeDependencies7.clear();
  	nodeDependencies7.push_back(kernelNode7);


	void* kernelArgs47[6] = {(void *)&d_input8, (void *)&d_output8, (void *)&gradient_h_output8, (void *)&gradient_v_output8, &width, &height};
	kernelNodeParams07.func = (void *)sobelFilter;
 	kernelNodeParams07.kernelParams = (void **)kernelArgs47;

	checkCudaErrors(
    cudaGraphAddKernelNode(&kernelNode8, graph, nodeDependencies8.data(),
                             nodeDependencies8.size(), &kernelNodeParams07));

  	nodeDependencies8.clear();
  	nodeDependencies8.push_back(kernelNode8);

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
	
	memcpyParams02.srcPtr = make_cudaPitchedPtr(d_output3, memSize, 1, 1);
	memcpyParams02.dstPtr = make_cudaPitchedPtr(final3, memSize, 1, 1);
	memcpyParams02.extent = make_cudaExtent(memSize, 1, 1);
	memcpyParams02.kind = cudaMemcpyDeviceToHost;

	checkCudaErrors(
      cudaGraphAddMemcpyNode(&memcpyNode3, graph, &kernelNode3,
                             1, &memcpyParams02));

	memcpyParams03.srcPtr = make_cudaPitchedPtr(d_output4, memSize, 1, 1);
	memcpyParams03.dstPtr = make_cudaPitchedPtr(final4, memSize, 1, 1);
	memcpyParams03.extent = make_cudaExtent(memSize, 1, 1);
	memcpyParams03.kind = cudaMemcpyDeviceToHost;

	checkCudaErrors(
      cudaGraphAddMemcpyNode(&memcpyNode4, graph, &kernelNode4,
                             1, &memcpyParams03));

	memcpyParams04.srcPtr = make_cudaPitchedPtr(d_output5, memSize, 1, 1);
	memcpyParams04.dstPtr = make_cudaPitchedPtr(final5, memSize, 1, 1);
	memcpyParams04.extent = make_cudaExtent(memSize, 1, 1);
	memcpyParams04.kind = cudaMemcpyDeviceToHost;

	checkCudaErrors(
      cudaGraphAddMemcpyNode(&memcpyNode, graph, &kernelNode,
                             1, &memcpyParams));


	memcpyParams05.srcPtr = make_cudaPitchedPtr(d_output6, memSize, 1, 1);
	memcpyParams05.dstPtr = make_cudaPitchedPtr(final6, memSize, 1, 1);
	memcpyParams05.extent = make_cudaExtent(memSize, 1, 1);
	memcpyParams05.kind = cudaMemcpyDeviceToHost;

	checkCudaErrors(
      cudaGraphAddMemcpyNode(&memcpyNode6, graph, &kernelNode6,
                             1, &memcpyParams05));
	
	memcpyParams06.srcPtr = make_cudaPitchedPtr(d_output7, memSize, 1, 1);
	memcpyParams06.dstPtr = make_cudaPitchedPtr(final7, memSize, 1, 1);
	memcpyParams06.extent = make_cudaExtent(memSize, 1, 1);
	memcpyParams06.kind = cudaMemcpyDeviceToHost;

	checkCudaErrors(
      cudaGraphAddMemcpyNode(&memcpyNode7, graph, &kernelNode7,
                             1, &memcpyParams06));

	memcpyParams07.srcPtr = make_cudaPitchedPtr(d_output8, memSize, 1, 1);
	memcpyParams07.dstPtr = make_cudaPitchedPtr(final8, memSize, 1, 1);
	memcpyParams07.extent = make_cudaExtent(memSize, 1, 1);
	memcpyParams07.kind = cudaMemcpyDeviceToHost;

	checkCudaErrors(
      cudaGraphAddMemcpyNode(&memcpyNode8, graph, &kernelNode8,
                             1, &memcpyParams07));


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

	cudaFree(d_input3);
	cudaFree(d_output3);
	cudaFree(gradient_h_output3);
	cudaFree(gradient_v_output3);

	cudaFree(d_input4);
	cudaFree(d_output4);
	cudaFree(gradient_h_output4);
	cudaFree(gradient_v_output4);

	cudaFree(d_input5);
	cudaFree(d_output5);
	cudaFree(gradient_h_output5);
	cudaFree(gradient_v_output5);

	cudaFree(d_input6);
	cudaFree(d_output6);
	cudaFree(gradient_h_output6);
	cudaFree(gradient_v_output6);

	cudaFree(d_input7);
	cudaFree(d_output7);
	cudaFree(gradient_h_output7);
	cudaFree(gradient_v_output7);

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
	pgmwrite("../images/image-output_2g_apollonian_gasket.ascii.pgm", (void *)final2,width, height);
	pgmwrite("../images/image-output_3g_apollonian_gasket.ascii.pgm", (void *)final3,width, height);
	pgmwrite("../images/image-output_4g_apollonian_gasket.ascii.pgm", (void *)final4,width, height);
	pgmwrite("../images/image-output_5g_apollonian_gasket.ascii.pgm", (void *)final5,width, height);
	pgmwrite("../images/image-output_6g_apollonian_gasket.ascii.pgm", (void *)final6,width, height);
	pgmwrite("../images/image-output_7g_apollonian_gasket.ascii.pgm", (void *)final7,width, height);
	pgmwrite("../images/image-output_8g_apollonian_gasket.ascii.pgm", (void *)final8,width, height);
	cudaDeviceReset();
	
	return 0;


}

