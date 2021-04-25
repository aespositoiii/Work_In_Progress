from stacking_funcs_017 import import_describe, laplace_threshold, mask_blur, registration, image_sort, reg_comb
from sklearn.decomposition import PCA
import sys
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os
import time

t0 = time.time()

directory = '/Users/anthonyesposito/Pictures/macroni/Rosasite_w_Conacalcite/1/JPG/'
hist_thresh = 10


filenames, images, file_nums, histograms = import_describe(directory, hist_thresh=hist_thresh)

t1 = time.time()
print('Import:  ', t1-t0)

images_lap = np.zeros(images.shape[0:3])

norm = np.zeros(images.shape[0], dtype='uint8')

print(images.shape)

print(images_lap.shape)

for i in range(images.shape[0]):
    print(i)
    images_lap[i], norm[i] = laplace_threshold(images[i], thresh=15, norm_blur=11)
    images_lap[i] = mask_blur(images_lap[i], n_iter=50, norm=norm[i])