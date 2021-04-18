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
    
    histograms = np.zeros(256*3)

    for j in range(img.shape[2]):
        hist, bin_edges = np.histogram(img[:,:,j], bins=np.arange(257))
        histograms[(j*256):(j*256+256)] = hist

    mask[i] = blurred_laplace

    # Display Results
    print(i, ' ', filenames[i], '  time_iter: ', time.time()-t2, '  time_elapsed: ', time.time()-t0)
    cv.imshow('window', img[:,:,0])
    cv.waitKey(1)
    colors = ['b', 'g', 'r']
    plt.figure()
    for j in range(3):
        plt.plot(np.arange(256), histograms[(j*256):(j*256+256)], color=colors[j])
        plt.yscale('log')   
    plt.show(block=False)
    plt.pause(3)
    plt.close()
    
