from stacking_funcs_008 import get_images, laplace_threshold, mask_blur, registration
from sklearn.decomposition import PCA
import sys
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os
import time

t0 = time.time()

directory = '/Users/anthonyesposito/Pictures/macroni/Rosasite_w_Conacalcite/1/JPG/'

filenames, images, file_nums, mask, histograms = get_images(directory)

print('Import:  ', time.time()-t0)

n_comps = 2
color_channels = images.shape[3]

images_pca = np.zeros((len(filenames), images.shape, n_comps))
pca = PCA(n_components=n_comps)

for i in range(images[])
    


pca.fit(histograms)
#print(type(pca.components_), pca.components_.shape, pca.explained_variance_ratio_)

images_pca = pca.transform(histograms)
images_pca_first = images_pca[...,0]
images_pca_second = images_pca[...,1]
image_order = np.argsort(images_pca_first, axis=0)
image_ordered = np.zeros(image_order.shape).astype('int64')


print(images_pca_first.shape, images_pca_first.dtype, images_pca_first)
print(images_pca_second.shape, images_pca_second.dtype, images_pca_second)

for i in range(len(image_order)):
    image_ordered[i] = list(image_order).index(i)
    cv.imshow('window', images[image_ordered[i]])
    cv.waitKey(1000)

plt.figure()
plt.plot(file_nums, images_pca_first, 'o')

plt.show(block=False)
plt.pause(5)
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
    colors = ['b', 'g', 'r']
    plt.figure()
    for j in range(3):
        plt.plot(np.arange(256), histograms[i,(j*256):(j*256+256)], color=colors[j])
        plt.yscale('log')   
    plt.show(block=False)
    plt.pause(3)
    plt.close()'''
    


print('total time: ', time.time()-t0)