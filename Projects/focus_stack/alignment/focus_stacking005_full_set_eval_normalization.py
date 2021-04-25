from stacking_funcs_003 import get_images, laplace_threshold, mask_blur
import sys
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os
import time

directory = '/Users/anthonyesposito/Pictures/macroni/Rosasite_w_Conacalcite/1/JPG/'
1
images = get_images(directory)

align = []
mask = []
for i in range(len(images)):

    img = images[i]
    print(type(img), img.shape)
    cv.imshow('window', img)
    cv.waitKey(1000)
    laplace_align, max_blur  = laplace_threshold(img, thresh=30)
    blurred_laplace= mask_blur(laplace_align , 0, 255, n_iter=50)
    
    align[i] = align.append(laplace_align)
    mask[i] = mask.append(blurred_laplace)
    print(type(blurred_laplace))
    cv.imshow('window', img)
    cv.waitKey(1000)
    cv.imshow('window', laplace_align)
    cv.waitKey(1000)
    cv.imshow('window', blurred_laplace )
    cv.waitKey(1000)
