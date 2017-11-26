## SIFFT2.py
#
# This is an updated version of subImageFFT that fixes a few issues with that
# program.
# This program takes a blurry image as an input and takes the 2D FFT of small
# sub-regions of the image to evaluate the degree of blur in that region.
# 
# It is an early part of an ongoing project to use differences of degree of motion
# blur in an image as a way to detect distance to objects in a robot's vision
#
# Author:       JEL
# Started:      11-3-17
# Last Edit:    11-17-17
#
##

import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy import signal

# Set the dimensions of the sub-regions
rowDim = 10
colDim = 10

# Load the image in grayscale

img = cv2.imread('blur1105_2.jpg',0)

# Convert to float

img = np.float32(img)


# Get the dimensions of the image

(rows, cols) = img.shape

# Pad the image with zeros for easier processing

if rows%rowDim != 0:
    m = rows%rowDim
    img = np.concatenate((img, np.zeros((rowDim-m,cols), dtype = int)), axis = 0)
    (rows, cols) = img.shape

if cols%colDim != 0:
    m = cols%colDim
    img = np.concatenate((img, np.zeros((rows,colDim-m), dtype = int)), axis = 1)
    (rows, cols) = img.shape


# Loop through the image
i = 0 
j = 0
fftimg = np.zeros((rows,cols), dtype = float)

while i < rows and j < cols:

    while i < rows:

        # Create the subimage
        simg = img[i:i+rowDim,j:j+colDim]

        # Remove the mean

        simg = simg - np.mean(simg)

        # Perform FFT, shift the plot
        dft = cv2.dft(np.float32(simg), flags = cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)

        # Get the magnitude, take the log of the values
        mag = cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1])
        mag = mag+10**-10
        log = np.log(mag)

        # Insert result into fftimg
        fftimg[i:i+rowDim,j:j+colDim] = log

        i = i+rowDim

    j = j+colDim
    i = 0



plt.subplot(121)
plt.title('Original')
plt.imshow(img, cmap = 'gray')
plt.xticks([])
plt.yticks([])

plt.subplot(122)
plt.title('fft')
plt.imshow(fftimg, cmap = 'gray')
plt.xticks([])
plt.yticks([])

##plt.figure()
##plt.title('Original')
##plt.imshow(img, cmap = 'gray')
##plt.xticks([])
##plt.yticks([])
##
##plt.figure()
##plt.title('fft')
##plt.imshow(fftimg, cmap = 'gray')
##plt.xticks([])
##plt.yticks([])


plt.show()

