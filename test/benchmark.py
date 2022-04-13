import numpy as np
import matplotlib.pyplot as plt


number_of_kernels = np.array([1,2,4,8,
                    16,32,64,128,
                    256,512,1024])
# speedup_600x600 = [0.001322/0.001336, 0.002631/0.002367 , 0.005307/0.004236, 0.010241/0.007457, 
#             0.019379/0.014720, 0.038247/0.030283,  0.076750/0.059610, 0.152795/0.115868,
#             0.307809/0.235399, 0.614759/0.467235, 1.230668/1.036315]
# speedup_300x246 = [0.000392/0.000341, 0.000780/0.000592,  0.001551/0.001111,  0.003096/0.002133,
#             0.006092/0.004173, 0.011180/0.008304, 0.022018/0.017402, 0.044393/0.032971,
#             0.090130/0.067389, 0.177791/0.130693, 0.355876/0.261324]
f_non_graph = open('test/benchmark_non_graph.csv','r')
lines_non_graph = f_non_graph.read().split('\n')
f_graph = open('test/benchmark_graph.csv','r')
lines_graph = f_graph.read().split('\n') 
speedup = []
for i in range(1,len(lines_non_graph)):
    if (lines_non_graph[i] == ''):
        continue 
    non_graph_time = float(lines_non_graph[i].split(',')[1])
    graph_time = float(lines_graph[i].split(',')[1])
    speedup.append(non_graph_time/graph_time)
fig = plt.figure()
# plt.plot(number_of_kernels,speedup_600x600,'-o',label='$Image$ $Size$ : 600x600')
# plt.plot(number_of_kernels,speedup_300x246,'-o',label='$Image$ $Size$ : 300x246')
plt.plot(number_of_kernels, speedup, '-o')
plt.title("Graph vs Non-graph comparision for fused Sobel operator")
plt.xlabel('Number of Images')
plt.ylabel('Speedup')
# plt.legend()
plt.savefig('test/benchmark_single_image.pdf')