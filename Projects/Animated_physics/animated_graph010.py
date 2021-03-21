import matplotlib.pyplot as plt
from matplotlib import animation, rc
import matplotlib.image as mpimg
import numpy as np
import time

m = 41
h = 301
w = h
x1 = np.array(np.linspace(0,3,h))
imagelist = np.empty((m, h, w))

imagelist[0,:,:] = np.sin(np.pi*0/20)*np.sin(np.pi*x1)

image = imagelist[0,:,:]

fig,ax = plt.subplots(1,1)
im = ax.imshow(image)

for i in range(m):
    d = np.sin(np.pi*i/20)*np.sin(np.pi*x1)
    d = d[np.newaxis,:]
    d = np.tile(d, (d.shape[1],1))
    imagelist[i,:,:] = d*d.T       
    image = imagelist[i,:,:]
    print(image[50,50])
    im.set_data(image)
    fig.canvas.draw()
    plt.pause(.1)
