from stacking_funcs_014 import import_describe, laplace_threshold, mask_blur, registration, image_sort
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

t1 = time.time()
print('Import:  ', t1-t0)

order = image_sort(images=images, filenames=filenames, file_nums=file_nums, histograms=histograms, n_comps=3, color_channels=images.shape[3], hist_thresh=hist_thresh)

t2 = time.time()
print('Sort: ', t2-t1, '  Total: ', t2-t0)

