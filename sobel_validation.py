import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt

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
                x = x.flatten() * self.vertical_grad_filter.flatten()
                sum_x = x.sum()

                y = self.image[i - kernel_width: i + kernel_width + 1, j - kernel_width: j + kernel_width + 1]
                y = y.flatten() * self.horizontal_grad_filter.flatten()
                sum_y = y.sum()
        
                grad_[i - kernel_width][j - kernel_width] = sqrt(sum_x**2 + sum_y**2)
        self.image = grad_
        return self.image
        # loc_time = time.localtime(time.time())
        # m = str(loc_time.tm_year) + str(loc_time.tm_mon) + str(loc_time.tm_mday) + str(loc_time.tm_hour) + str(loc_time.tm_min) + str(loc_time.tm_sec)
        # img_save_name = 'sobel_edge_det_' + m + ".jpg"
        # plt.imsave(img_save_name, self.image)

# Load input image
input_user = readpgm('../images/image512x512.pgm')

user_img = np.reshape(input_user[0],input_user[1])

# Prepare the image blur matrix
b1 = np.array([[0.0,0.2,0.0],[0.2,0.2,0.2],[0.0,0.2,0.0]])

# Apply the image operator
user_img = convolve2d(user_img, b1, "same", "symm")

img = sobel_edge_detector(user_img)
G = img.detect_edges()

# # Prepare the sobel kernels
# a1 = np.matrix([1, 2, 1])
# a2 = np.matrix([-1, 0, 1])
# Kx = a1.T * a2
# Ky = a2.T * a1

# # Apply the Sobel operator
# Gx = convolve2d(user_img, Kx, "same", "symm",fillvalue=0)
# Gy = convolve2d(user_img, Ky, "same", "symm",fillvalue=0)
# G = np.sqrt(Gx*Gx + Gy*Gy)

# Rotate the image
import scipy.ndimage as ndimage

angle = 90 # in degrees

rotated_G = ndimage.rotate(G, angle, reshape=True)

# Flip the image Left-Right
from PIL import Image
from PIL import ImageOps

im = Image.fromarray(rotated_G)

im = ImageOps.mirror(im)

im = np.asarray(im)

im_bool = im > 30

plt.imshow(im_bool*255*np.ones(im.shape))


plt.savefig("theoretical.png")

input_kernel = readpgm('../images/image-output_ng_512x512.pgm')
kernel_img = np.reshape(input_kernel[0],input_kernel[1])

plt.imshow(kernel_img)
plt.savefig("kernel_ng.png")

input_kernel = readpgm('../images/image-output_g_512x512.pgm')
kernel_img = np.reshape(input_kernel[0],input_kernel[1])

plt.imshow(kernel_img)
plt.savefig("kernel_g.png")
