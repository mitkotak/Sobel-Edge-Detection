import numpy as np
import matplotlib.pyplot as plt
size = np.array([256*256, 1024*1024, 2048*2048, 4096*4096, 16384*16384])
ng_parallel_time = np.array([0.000782,0.009875,0.017384,0.083358,0.756567])
ng_serial_time = np.array([0.061355,0.313938,1.080509,5.889417,78.341225])
ng_total_time = np.array([0.062136,0.323813,1.097893,5.972775,79.097794])

g_parallel_time = np.array([0.000472,0.001092,0.001971,0.005267,0.022165])
g_serial_time = np.array([0.149637,0.279581,1.253908,5.898839,77.035622])
g_total_time = np.array([0.150109,0.280673,1.255879,5.904107,77.057793])

print(ng_total_time/g_total_time)