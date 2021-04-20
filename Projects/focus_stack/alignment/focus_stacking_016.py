from stacking_funcs_008 import get_images, laplace_threshold, mask_blur, registration
from sklearn.decomposition import PCA
import sys
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os
import time

t0 = time.time()

directory = '/Users/anthonyesposito/Pictures/macroni/Rosasite_w_Conacalcite/5/JPG/'

filenames, images, file_nums, mask, histograms = get_images(directory, hist_thresh=10)

print('Import:  ', time.time()-t0)

n_comps = 3
color_channels = images.shape[3]

images_pca = np.zeros((len(filenames), color_channels*n_comps))
pca = PCA(n_components=n_comps)

colors = ['b', 'g', 'r']
markers = ['.', 'x', '+']
plt.figure()

for i in range(color_channels):
    pca.fit(histograms[:,(i*256):(i*256+256)])
    pca_temp = pca.transform(histograms[:,(i*256):(i*256+256)])
    for j in range(n_comps):
        pca_temp_ij = pca_temp[...,j]
        pca_temp_ij_min = pca_temp_ij.min()
        pca_temp_ij_max = pca_temp_ij.max()
        images_pca[...,i*color_channels+j] = ( pca_temp_ij - pca_temp_ij_min ) / pca_temp_ij_max
        plt.plot(file_nums, images_pca[:,i*color_channels+j], markers[j], color=colors[i])

print(images_pca.shape)

#print(type(pca.components_), pca.components_.shape, pca.explained_variance_ratio_)
'''
image_order = np.argsort(images_pca_first, axis=0)
image_ordered = np.zeros(image_order.shape).astype('int64')

for i in range(len(image_order)):
    image_ordered[i] = list(image_order).index(i)
    cv.imshow('window', images[image_ordered[i]])
    cv.waitKey(1000)
'''

plt.show(block=False)
plt.pause(20)
plt.close()

t1 = time.time()

for i in range(len(images)):

    t2 = time.time()

    img = images[i]
    laplace_align, max_blur  = laplace_threshold(img, thresh=15)
    blurred_laplace= mask_blur(laplace_align, max_blur, n_iter=50)

    mask[i] = blurred_laplace

    # Display Results
    '''print(i, ' ', filenames[i], '  time_iter: ', time.time()-t2, '  time_elapsed: ', time.time()-t0)
    cv.imshow('window', img[:,:,0])
    cv.waitKey(1)
    
    plt.figure()
    for j in range(3):
        plt.plot(np.arange(256), histograms[i,(j*256):(j*256+256)], color=colors[j])
        plt.yscale('log')   
    plt.show(block=False)
    plt.pause(3)
    plt.close()'''
    


print('total time: ', time.time()-t0)