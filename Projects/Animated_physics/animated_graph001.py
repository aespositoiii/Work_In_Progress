import matplotlib.pyplot as plt
from matplotlib import animation, rc
import matplotlib.image as mpimg
import numpy as np
import time

i = 10
h = 256
w = h
x1 = np.array(np.linspace(0,3,h))
imagelist = np.empty((i, h, w))
imagelist[0, :, :] = np.sin(np.pi*0/25)*np.sin(np.pi*x1)
fig, ax = plt.subplots(1,1)

im = ax.imshow(imagelist[0, :, :], cmap='hot')

for i in range(i):
    d = np.sin(np.pi*i/25)*np.sin(np.pi*x1)
    d = d[np.newaxis,:]
    d = np.tile(d, (d.shape[1],1))
    imagelist[i,:,:] = d*d.T
    im.set_data(imagelist[i, :, :])
    fig.canvas.draw()
    plt.pause(.5)


