# Sobel_Edge_Detection
Implemented sobel edge detection using MPI and Cuda

**REQUIREMENTS**
1. MPI (http://mpitutorial.com/tutorials/installing-mpich2/)
2. CUDA or run on Google Colab (https://medium.com/@iphoenix179/running-cuda-c-c-in-jupyter-or-how-to-run-nvcc-in-google-colab-663d33f53772) 

The code only works for images with '.pgm' extension.
'pgmio.h' is a helper file to read and write the image.

**For MPI**, 
In the main.c file, 
1. From line 11 to line 23, sample image sizes are given. Select which image you want to use for edge detection and comment out the other two.
2. From line 55 to line 57, input image file names are specified. Use the required files, and comment out the others.
3. From line 115 to line 117, output image file names are specified. Use the required files, and comment out the others.

To compile the code do:
  $MPICC -o main main.c -lm
  
To run the code do:
  $MPIRUN -n 4 ./main
  
Here -n is the number of processes you want to use to parallelize the code.

**For Cuda**
In the main.cu file,
1. In lines 18 and 19, sample image size is given. Change the value if you wish to use a different image.
2. A function load_image is provided. Change the input image file name if you wish to use a different image. 
3. A function save_image is provided. Change the output image file name if you wish to use a different image. 

After several rounds of trial and error, we set the threshold value as 100, as it gave the most ideal result. It can be changed in in the main.c or main.cu.
