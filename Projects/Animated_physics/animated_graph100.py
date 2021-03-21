import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import time

m = 101
h = 301
w = h
x1 = np.array(np.linspace(0,3,h))

imagelist = np.empty((m, h, w))
imagelist[0,:,:] = np.sin(np.pi*0/20)*np.sin(np.pi*x1)

fig,ax = plt.subplots(1,1)
d = np.sin(np.pi*x1)
d = d[np.newaxis,:]
d = np.tile(d, (d.shape[1],1))
image1 = d*d.T
im = ax.imshow(image1, cmap='jet')



plt.colorbar(im)

for i in range(m):    
    d = np.sin(np.pi*i/20)*np.sin(np.pi*x1 - np.pi*i/10)
    d = d[np.newaxis,:]
    d = np.tile(d, (d.shape[1],1))
    imagelist[i,:,:] = d*d.T       
    image = imagelist[i,:,:]
    im.set_data(image)
    print(i)
    fig.canvas.draw_idle()
    plt.pause(.0001)
