all: run

run: sobel_non_graph sobel_graph benchmark_py

sobel_graph: compile_sobel_graph
	./a_graph

sobel_non_graph: compile_sobel_non_graph
	./a_non_graph

compile_sobel_graph: sobel_graph.cu
	nvcc -I${PWD} -g -G sobel_graph.cu -o a_graph

compile_sobel_non_graph: sobel_non_graph.cu
	nvcc -I${PWD} -g -G sobel_non_graph.cu -o a_non_graph

benchmark_py: test/benchmark.py
	python3 test/benchmark.py