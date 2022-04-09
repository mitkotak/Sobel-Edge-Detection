import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import numpy.linalg as la

def readpgm(name):
    with open(name) as f:
        lines = f.readlines()

    # Ignores commented lines
    for l in list(lines):
        if l[0] == '#':
            lines.remove(l)

    # Makes sure it is ASCII format (P2)
    assert lines[0].strip() == 'P2' 

    # Converts data to a list of integers
    data = []
    for line in lines[1:]:
        data.extend([int(c) for c in line.split()])

    return (np.array(data[3:]),(data[1],data[0]),data[2])


import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
from math import sqrt
import time

class sobel_edge_detector:
    def __init__(self, img):
        self.image = img
        self.vertical_blur_filter =  np.array([[0.0,0.2,0.0],[0,0.2,0],[0.0,0.2,0.0]])
        self.horizontal_blur_filter = np.array([[0.0,0,0.0],[0.2,0,0.2],[0.0,0,0.0]])
        self.vertical_grad_filter = np.array([[1, 0, -1],[2, 0, -2],[1, 0, -1]])
        self.horizontal_grad_filter = np.array([[1, 2, 1],[0, 0, 0],[-1, -2, -1]])

    def detect_edges(self):
        kernel_width = self.vertical_grad_filter.shape[0]//2
        grad_ = np.zeros(self.image.shape)

        self.image = np.pad(self.image, pad_width= ([kernel_width, ], [kernel_width, ]), 
        mode= 'constant', constant_values= (0, 0))
        for i in range(kernel_width, self.image.shape[0] - kernel_width):
            for j in range(kernel_width, self.image.shape[1] - kernel_width):
                
                x = self.image[i - kernel_width: i + kernel_width + 1, j - kernel_width: j + kernel_width + 1]
                x = (x.flatten() * self.vertical_grad_filter.flatten()).astype(np.int32)
                sum_x = int(x.sum())

                y = self.image[i - kernel_width: i + kernel_width + 1, j - kernel_width: j + kernel_width + 1]
                y = (y.flatten() * self.horizontal_grad_filter.flatten()).astype(np.int32)
                sum_y = int(y.sum())
        
                grad_[i - kernel_width][j - kernel_width] = sqrt(sum_x**2 + sum_y**2)
        self.image = grad_
        return self.image


width = 600
height = 600
# Load input image
input_user = readpgm('images/apollonian_gasket.ascii.pgm')
user_img = np.reshape(input_user[0],input_user[1])
G = user_img.copy()

plt.imshow(user_img)
plt.savefig("user.png")


blur = np.array([[0.0,0.2,0.0],[0.2,0.2,0.2],[0.0,0.2,0.0]])

Kx =  np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
Ky =  np.array([[-1,-2,-1],[0,0,0],[1,2,1]])

blur_user_img = user_img.copy()

for x in range(1,width-1):
    for y in range(1,height-1):
        blur_temp = int(blur[0][0]*user_img[x-1][y-1]) \
                    + int(blur[0][1]*user_img[x][y-1]) \
                    + int(blur[0][2]*user_img[x+1][y-1]) \
                    + int(blur[1][0]*user_img[x-1][y]) \
                    + int(blur[1][1]*user_img[x][y]) \
                    + int(blur[1][2]*user_img[x+1][y]) \
                    + int(blur[2][0]*user_img[x-1][y+1]) \
                    + int(blur[2][1]*user_img[x][y+1]) \
                    + int(blur[2][2]*user_img[x+1][y+1])
        
        blur_user_img[x][y] = blur_temp

for x in range(1,width-1):
    for y in range(1,height-1):
        pixel_x = int(Kx[0][0]*blur_user_img[x-1][y-1]) \
                    + int(Kx[0][1]*blur_user_img[x][y-1]) \
                    + int(Kx[0][2]*blur_user_img[x+1][y-1]) \
                    + int(Kx[1][0]*blur_user_img[x-1][y]) \
                    + int(Kx[1][1]*blur_user_img[x][y]) \
                    + int(Kx[1][2]*blur_user_img[x+1][y]) \
                    + int(Kx[2][0]*blur_user_img[x-1][y+1]) \
                    + int(Kx[2][1]*blur_user_img[x][y+1]) \
                    + int(Kx[2][2]*blur_user_img[x+1][y+1])
        
        pixel_y = int(Ky[0][0]*blur_user_img[x-1][y-1]) \
                    + int(Ky[0][1]*blur_user_img[x][y-1]) \
                    + int(Ky[0][2]*blur_user_img[x+1][y-1]) \
                    + int(Ky[1][0]*blur_user_img[x-1][y]) \
                    + int(Ky[1][1]*blur_user_img[x][y]) \
                    + int(Ky[1][2]*blur_user_img[x+1][y]) \
                    + int(Ky[2][0]*blur_user_img[x-1][y+1]) \
                    + int(Ky[2][1]*blur_user_img[x][y+1]) \
                    + int(Ky[2][2]*blur_user_img[x+1][y+1])

        val = sqrt((pixel_x * pixel_x) + (pixel_y * pixel_y))

        if(val < 30):
            val = 0
        else:
            val = 255

        G[x][y] = val

print(G.shape)

# Apply the image operator
# user_img = convolve2d(user_img, b1, "same", "symm")

# img = sobel_edge_detector(user_img)
# G = img.detect_edges()

# # Prepare the sobel kernels
# a1 = np.matrix([1, 2, 1])
# a2 = np.matrix([-1, 0, 1])
# Kx = a1.T * a2
# Ky = a2.T * a1

# # Apply the Sobel operator
# Gx = convolve2d(user_img, Kx, "same", "symm",fillvalue=0)
# Gy = convolve2d(user_img, Ky, "same", "symm",fillvalue=0)
# G = np.sqrt(Gx*Gx + Gy*Gy)



theoretical_img = G
plt.imshow(theoretical_img)
plt.savefig("theoretical.png")

input_kernel = readpgm('images/image-output_ng_apollonian_gasket.ascii.pgm')
kernel_ng_img = np.reshape(input_kernel[0],input_kernel[1])

plt.imshow(kernel_ng_img)
plt.savefig("kernel_ng.png")

input_kernel = readpgm('images/image-output_g_apollonian_gasket.ascii.pgm')
kernel_g_img = np.reshape(input_kernel[0],input_kernel[1])

plt.imshow(kernel_g_img)
plt.savefig("kernel_g.png")

print("2 norm of graph and non-graph ",la.norm(kernel_g_img - kernel_ng_img,2))
print("2 norm of graph and theoretical ",la.norm(kernel_g_img - theoretical_img,2))
print("2 norm of non-graph and theoretical ",la.norm(kernel_ng_img - theoretical_img,2))
