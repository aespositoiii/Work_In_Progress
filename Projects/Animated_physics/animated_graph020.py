import matplotlib.pyplot as plt
from matplotlib import animation, rc
import matplotlib.image as mpimg
import numpy as np
import time

m = 41
h = 301
w = h
x1 = np.array(np.linspace(0,3,h))

fig,ax = plt.subplots(1,1)
d = np.sin(np.pi*x1)
d = d[np.newaxis,:]
d = np.tile(d, (d.shape[1],1))
image1 = d*d.T
im = ax.imshow(image1, cmap='prism')

plt.colorbar(im)

for i in range(2001):    
    image = np.multiply(np.sin(np.pi*i/25), image1)
    im.set_data(image)
    print(i, image.shape)
    fig.canvas.draw_idle()
    plt.pause(.02)

    
