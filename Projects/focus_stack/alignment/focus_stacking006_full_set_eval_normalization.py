from stacking_funcs_004 import get_images, laplace_threshold, mask_blur
import sys
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os
import time

directory = '/Users/anthonyesposito/Pictures/macroni/Rosasite_w_Conacalcite/1/JPG/'

images, align, mask = get_images(directory)


for i in range(len(images)):

    img = images[i]
    cv.imshow('window', img)
    cv.waitKey(1000)
    laplace_align, max_blur  = laplace_threshold(img, thresh=15)
    cv.imshow('window', laplace_align)
    cv.waitKey(1000)
    blurred_laplace= mask_blur(laplace_align, max_blur, n_iter=50)
    
    align[i] = laplace_align
    mask[i] = blurred_laplace