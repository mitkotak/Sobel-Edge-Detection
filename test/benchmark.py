import numpy as np
import matplotlib.pyplot as plt

number_of_kernels = 5*np.array([1,2,4,8,
                    16,32,64,128,
                    256,512,1024])
speedup = [0.001322/0.001336, 0.002631/0.002367 , 0.005307/0.004236, 0.010241/0.007457, 
            0.019379/0.014720, 0.038247/0.030283,  0.076750/0.059610, 0.152795/0.115868,
            0.307809/0.235399, 0.614759/0.467235, 1.230668/1.036315]
plt.plot(np.log(number_of_kernels),speedup,'-o')
plt.title("Graph vs Non-graph comparision for fused Sobel operator")
plt.xlabel('log(Number of kernels)')
plt.ylabel('Speedup')
plt.savefig('benchmark.png')