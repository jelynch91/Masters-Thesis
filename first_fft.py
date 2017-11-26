# This is a first attempt at generating the 2D Fourier
# transform of an image using OpenCV

import numpy as np
import cv2
from matplotlib import pyplot as plt

mask = np.zeros((6,6), dtype=int)


# Loads the image
img = cv2.imread('motion_bike.jpg',0)

# Performs fft (after converting to float32 and handling the complex output
# Also shifts the output such that 0 is at the center
dft = cv2.dft(np.float32(img), flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)


mag = cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1])

# Mask the center to elimate some of the dominating values at low frequencies

mag_r = mag.shape[0]
mag_c = mag.shape[1]
ctr = [mag_r//2,mag_c//2]

#mag[ctr[0]-3:ctr[0]+3,ctr[1]-3:ctr[[1]+3] = mask 

# Converts values to decibels
magnitude_spectrum = 20*np.log(mag)


plt.subplot(131),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])

plt.subplot(132),plt.imshow(mag, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])


plt.subplot(133),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum (dB)'), plt.xticks([]), plt.yticks([])


plt.show()
