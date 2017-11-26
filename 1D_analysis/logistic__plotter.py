import numpy as np
from matplotlib import pyplot as plt

x = np.arange(-20,60,0.1)

k = [0.1, 0.5, 2, 10]

j=1

plt.figure("Logistic Functions")
plt.title("Logistic Functions")

for i in k:

    y = 1/(1+1*np.exp(-i*x))


    plt.subplot(2,4,j)
    plt.title("K = {}".format(i))
    plt.plot(x,y)
    j += 1

for i in k:

    y = 1/(1+1*np.exp(-i*x))
    y = y - np.mean(y)
    fft = np.fft.fft(y)
    freq = np.fft.fftfreq(x.shape[-1], 0.01)

    plt.subplot(2,4,j)
    plt.title("FFT (K = {})".format(i))
    plt.plot(freq.real[0:100],np.abs(fft[0:100]))
    j+=1

plt.show()
