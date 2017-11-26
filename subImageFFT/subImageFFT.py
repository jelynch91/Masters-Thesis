## subImageFFT.py
#
# This program takes a blurry image as an input and takes the 2D FFT of small
# sub-regions of the image to evaluate the degree of blur in that region.
#
# It is an early part of an ongoing project to use differences of degree of motion
# blur in an image as a way to detect distance to objects in a robot's vision
#
# Author:       JEL
# Started:      10-29-17
# Last Edit:    10-29-17
#
##

import numpy as np
import cv2
from matplotlib import pyplot as plt

# Set the dimensions of the sub-regions and create a null list for images
rowdim = 16
coldim = 16
chunks = []

# Load the image in grayscale

img = cv2.imread('motion_bike.jpg',0)

# Get the dimensions of the image

(rows, cols) = img.shape

# Loop through the image
i = 0
j = 0

while (i+rowdim <= rows) and (j+coldim <= cols):

    while j+coldim <= cols:

        # Create the subimage
        simg = img[i:i+rowdim,j:j+coldim]

        # Perform FFT, shift the plot
        dft = cv2.dft(np.float32(simg), flags = cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)

        # Get the magnitude, take the log of the values
        mag = cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1])
        log = np.log(mag)

        # Store log in chunks

        chunks.append(log)
        l = len(chunks)
        # Iterate along the columns
        j = j+coldim+1

    # When you get to the end of the columns, go back to the beginning
    # Iterate along the rows
    j = 0
    i = i+rowdim+1


# Take the list of subregions and reconstitute it into a new frequency image
# Use np.concatenate to create columns of the elements of chunks. Then use
# it to combine the colums of elements to create the new image

k = 1
newrow = chunks[0]
rowList = []

 

newrow = np.concatenate((chunks), axis = 1)
 
c = 450
r = 10
n = 1
m =0

newim = newrow[0:rowdim,0:c]

while m < 28:
    newim = np.concatenate((newim,newrow[0:r,n*(c+1):n*(c+1)+c]), axis = 0)
    m +=1
    n +=1





plt.subplot(121)
plt.title('Original')
plt.imshow(img, cmap = 'gray')
plt.xticks([])
plt.yticks([])

##plt.subplot(234)
##plt.title('10pX10p slice')
##plt.imshow(chunks[1349], cmap = 'gray')
##plt.xticks([])
##plt.yticks([])

plt.subplot(122)
plt.title('FFT over Small Square Regions')
plt.imshow(newim, cmap = 'gray')
plt.xticks([])
plt.yticks([])



plt.show()


    

