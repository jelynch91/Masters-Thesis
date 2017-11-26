## 1DFreqDist.py
#
# This code borrows from SIFFT2.py to take in an image, break it into parts,
# and take the 2D fft of each part. In addition, I define a function getRadial,
# which takes the 2D fft and sums about equal radii to get a 1D representation
# of the frequency distribution. This will be used later to estimate blur in
# that region of the image.
#
# It is an early part of an ongoing project to use differences of degree of motion
# blur in an image as a way to detect distance to objects in a robot's vision
#
# Author:       JEL
# Started:      11-20-17
# Last Edit:    11-20-17
#
##

import numpy as np
import cv2
from matplotlib import pyplot as plt

#############################################################################
# getRadial
#
# Takes a square array as an input and outputs a 1D array that gives the sum
# over the values that are equidistant from the center of the image.
#
#############################################################################

def getRadial(A):

    # Initialize the counters
    i=0
    j=0
    blur = 0
    # This gives half the side length of the square matrix
    d= A.shape[0]/2

    mask = np.zeros([10,10])
    rTemp = []

    # Loop through the array. When the radius of a location matches k, add it
    # to the temporary list. Find the mean of the values in the list, Then
    # assign it to rArray


    while i<A.shape[0]:
        while j<A.shape[1]:
            x = j-d
            y = i-d
            r = np.int(np.sqrt(x**2+y**2))
            if (r>5):
                rTemp.append(A[i,j])

            j+=1
        i+=1
        j=0


    blur = np.mean(rTemp)

    mask = mask+blur

    return mask

##############################################################################
# Main Body


# Set the dimensions of the sub-regions
rowDim = 10
colDim = 10

# Load the image in grayscale

img = cv2.imread('motion_bike.jpg',0)

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
radList = []

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

        # Eliminate any zero values
        mag = mag+10**-10

        # Get the logarithm of the magnitude image
        log = np.log(mag)

        # get the radial distribution array and add it to the list of other
        # arrays
        mask = getRadial(log)

        # Insert result into fftimg
        fftimg[i:i+rowDim,j:j+colDim] = mask

        i = i+rowDim

    j = j+colDim
    i = 0


# Plot the
plt.figure()
plt.subplot(121)
plt.title('Original')
plt.imshow(img, cmap = 'gray')
plt.xticks([])
plt.yticks([])

plt.subplot(122)
plt.title('Blur Estimate')
plt.imshow(fftimg, cmap = 'gray')
plt.xticks([])
plt.yticks([])


plt.show()
