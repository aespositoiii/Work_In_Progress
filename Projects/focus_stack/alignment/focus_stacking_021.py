from stacking_funcs_013 import import_describe, laplace_threshold, mask_blur, registration, image_sort
from sklearn.decomposition import PCA
import sys
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os
import time

t0 = time.time()

directory = '/Users/anthonyesposito/Pictures/macroni/Rosasite_w_Conacalcite/5/JPG/'
hist_thresh = 10


filenames, images, file_nums, mask, histograms = import_describe(directory, hist_thresh=hist_thresh)

print('Import:  ', time.time()-t0)

order = image_sort(images=images, filenames=filenames, file_nums=file_nums, histograms=histograms, n_comps=3, color_channels=images.shape[3], hist_thresh=hist_thresh)

# stacked = focus_stack(images, order)

#print(type(pca.components_), pca.components_.shape, pca.explained_variance_ratio_)

t1 = time.time()

colors = ['b', 'g', 'r']

hist_width = 256-hist_thresh

for i in range(len(images)):

    t2 = time.time()

    img = images[i]
    laplace_align, max_blur  = laplace_threshold(img, thresh=15)
    blurred_laplace= mask_blur(laplace_align, max_blur, n_iter=50)

    mask[i] = blurred_laplace

    # Display Results
    print(i, ' ', filenames[i], '  time_iter: ', time.time()-t2, '  time_elapsed: ', time.time()-t0)
    cv.imshow('window', img[:,:,0])
    cv.waitKey(1)
    
    plt.figure()
    for j in range(3):
        plt.plot(np.arange(hist_width), histograms[i,(j*hist_width):(j*hist_width+hist_width)], color=colors[j])
        plt.yscale('log')   
    plt.show(block=False)
    plt.pause(3)
    plt.close()

print('total time: ', time.time()-t0)