from stacking_funcs_008 import get_images, laplace_threshold, mask_blur, registration
from sklearn.decomposition import PCA
import sys
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os
import time

t0 = time.time()

directory = '/Users/anthonyesposito/Pictures/macroni/Rosasite_w_Conacalcite/1/check/JPG/'

filenames, images, file_nums, mask, histograms = get_images(directory)

print('Import:  ', time.time()-t0)

pca = PCA(n_components=2)
pca.fit(histograms)
print(type(pca.components_), pca.components_.shape, pca.explained_variance_ratio_)

images_pca = pca.transform(histograms)
plt.figure()
plt.plot(file_nums, images_pca, 'o')
plt.show(block=False)
plt.pause(5)
plt.close



t1 = time.time()

for i in range(len(images)):

    t2 = time.time()

    img = images[i]
    laplace_align, max_blur  = laplace_threshold(img, thresh=15)
    blurred_laplace= mask_blur(laplace_align, max_blur, n_iter=50)

    mask[i] = blurred_laplace

    # Display Results
    print(i, ' ', filenames[i], '  time_iter: ', time.time()-t2, '  time_elapsed: ', time.time()-t0)
    '''cv.imshow('window', img[:,:,0])
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