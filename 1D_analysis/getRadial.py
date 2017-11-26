import numpy as np
import cv2
from matplotlib import pyplot as plt

dim = 10


img = cv2.imread('motion_bike.jpg',0)

trim = img[0:dim,0:dim]
trim = trim - np.mean(trim)

dft = cv2.dft(np.float32(trim), flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

mag = cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1])

i=0
j=0
k=0

# This gives half the side length of the square matrix
d= mag.shape[0]/2

# The maximum value of r that we'll find
rmax = np.int(np.sqrt(d**2+d**2))
rArray = np.zeros((rmax+1), dtype = float)
rTemp = []

while (k<rmax+1):
    while i<mag.shape[0]:
        while j<mag.shape[1]:
            x = j-d
            y = i-d
            r = np.int(np.sqrt(x**2+y**2))
            if (r==k):
                rTemp.append(mag[i,j])

            j+=1
        i+=1
        j=0


    rArray[k]= np.mean(rTemp)
    k+=1
    i=0
    j=0
    rTemp = []

print rTemp
print rArray

adj_rArray = np.divide(rArray,np.max(rArray))

plt.plot(rArray)

plt.show()
