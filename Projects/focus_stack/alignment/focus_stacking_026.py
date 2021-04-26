from stacking_funcs_018 import import_describe, laplace_threshold, mask_blur, registration, image_sort, reg_comb
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


filenames, images, file_nums, mask, histograms = import_describe(directory, hist_thresh=hist_thresh)

t1 = time.time()
print('Import:  ', t1-t0)

order, trans_on = image_sort(images=images, filenames=filenames, file_nums=file_nums, histograms=histograms, n_comps=3, color_channels=images.shape[3], hist_thresh=hist_thresh)

t2 = time.time()
print('Sort: ', t2-t1, '  Total: ', t2-t0)

comb = reg_comb(images, order, trans_on, file_nums, thresh=30, norm_blur=13, n_iter=50, exp=3, warp_mode=cv.MOTION_EUCLIDEAN, number_of_iterations=1000, termination_eps=1e-3)

print(comb)

cv.imwrite((directory + 'test' + '.jpg'), comb)