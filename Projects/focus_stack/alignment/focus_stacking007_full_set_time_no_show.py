from stacking_funcs_005_no_show import get_images, laplace_threshold, mask_blur
import sys
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os
import time

t0 = time.time()

directory = '/Users/anthonyesposito/Pictures/macroni/Rosasite_w_Conacalcite/1/JPG/'

images, align, mask = get_images(directory)

print('Import:  ', time.time()-t0)

t1 = time.time()
for i in range(len(images)):

    t2 = time.time()

    img = images[i]
    laplace_align, max_blur  = laplace_threshold(img, thresh=15)
    blurred_laplace= mask_blur(laplace_align, max_blur, n_iter=50)
    

    align[i] = laplace_align
    mask[i] = blurred_laplace

    print(i, '  time_iter: ', time.time()-t2, '  time_elapsed: ', time.time()-t0)