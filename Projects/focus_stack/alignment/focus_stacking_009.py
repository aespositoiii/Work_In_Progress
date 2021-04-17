from stacking_funcs_007 import get_images, laplace_threshold, mask_blur, registration
import sys
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os
import time

t0 = time.time()

directory = '/Users/anthonyesposito/Pictures/macroni/Rosasite_w_Conacalcite/1/JPG/'

filenames, images, mask = get_images(directory)

print('Import:  ', time.time()-t0)

t1 = time.time()
for i in range(len(images)):

    t2 = time.time()

    img = images[i]
    laplace_align, max_blur  = laplace_threshold(img, thresh=15)
    blurred_laplace= mask_blur(laplace_align, max_blur, n_iter=50)
    
    histograms = np.zeros(25)

    mask[i] = blurred_laplace

    print(i, ' ', filenames[i], '  time_iter: ', time.time()-t2, '  time_elapsed: ', time.time()-t0)
    
    cv.imshow('window', img[:,:,0])
    cv.waitKey(1)
    colors = ['b', 'g', 'r']
    plt.figure()
    for j in range(3):
        plt.hist(img[:,:, j].ravel(), 256,[0,255], color=colors[j], alpha=.5)
        plt.yscale('log')   
    plt.show(block=False)
    plt.pause(3)
    plt.close()
    
