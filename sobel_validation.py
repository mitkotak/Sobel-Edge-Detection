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


# Load input image
input_user = readpgm('image512x512.pgm')

user_img = np.reshape(input_user[0],input_user[1])

# Prepare the image blur matrix
b1 = np.array([[0.0,0.2,0.0],[0.2,0.2,0.2],[0.0,0.2,0.0]])

# Apply the image operator
user_img = convolve2d(user_img, b1, "same", "symm")

# Prepare the sobel kernels
a1 = np.matrix([1, 2, 1])
a2 = np.matrix([-1, 0, 1])
Kx = a1.T * a2
Ky = a2.T * a1

# Apply the Sobel operator
Gx = convolve2d(user_img, Kx, "same", "symm")
Gy = convolve2d(user_img, Ky, "same", "symm")
G = np.sqrt(Gx**2 + Gy**2)

input_kernel = readpgm('image-outputl512x512.pgm')
kernel_img = np.reshape(input_kernel[0],input_kernel[1])

plt.imshow(kernel_img)
plt.savefig("kernel.png")

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