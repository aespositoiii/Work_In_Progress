from stacking_funcs_033b import import_describe, laplace_threshold, mask_blur, registration, image_sort, reg_comb
from sklearn.decomposition import PCA
import sys
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os
import time

directory = '/Users/anthonyesposito/Pictures/macroni/5_4_2021_test/JPG/'
destination = 'stacked'
ext = '.jpg'
hist_min = 0
hist_max = 255
n_comps = 3
thresh = 40
norm_blur=11
n_iter=10
exp=3
warp_mode=cv.MOTION_EUCLIDEAN
number_of_iterations=1000
termination_eps=1e-3

t0 = time.time()

filenames, images, file_nums, lap_mask, histograms, transformed = import_describe(directory, ext, hist_min=hist_min, hist_max=hist_max)

t1 = time.time()
print('Import:  ', t1-t0)
if transformed == False:
    order, trans_on = image_sort(images=images, filenames=filenames, file_nums=file_nums, histograms=histograms, n_comps=n_comps, color_channels=images.shape[3], hist_min=hist_min, hist_max=hist_max)
else:
    order = None
    trans_on = None
t2 = time.time()
print('Sort: ', t2-t1, '  Total: ', t2-t0)

comb = reg_comb(images, order, trans_on, file_nums, transformed=transformed, ext=ext, directory=directory, lap_mask=lap_mask, thresh=thresh, norm_blur=norm_blur, n_iter=n_iter, exp=exp, warp_mode=warp_mode, number_of_iterations=number_of_iterations, termination_eps=termination_eps)

cv.imwrite((directory + destination + '/test' + str(int(time.time())) + '.jpg'), comb)