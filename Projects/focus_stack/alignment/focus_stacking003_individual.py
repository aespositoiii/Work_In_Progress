from stacking_funcs import get_images, laplace_threshold, mask_blur
import sys
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os
import time

directory = '/Users/anthonyesposito/Pictures/macroni/Rosasite_w_Conacalcite/1/JPG/'

images = get_images(directory)

'''
for i in range(len(images)):
    cv.imshow('window', images[i])
    cv.waitKey(100)
'''

'''
for i in range(2,50):
    blurred_align = mask_blur(blurred_align, i)
'''
"""
for i in range(len(images)):

    img = images[i]
    laplace_align, max_blur  = laplace_threshold(img, thresh=15)
    
    '''
    blurred_align = mask_blur(laplace_align, j=1)
    for n in range(50):
        blurred_align = mask_blur(blurred_align, n)
    '''

    print(i)
    print('a')
    #cv.imshow('window', img)
    #cv.waitKey(1000)
    print(i, ' b ', max_blur)
    #cv.imshow('window', laplace_align)
    #cv.waitKey(1000)
    
    
    print('c')
    cv.imshow('window', blurred_align)
    cv.waitKey(1000)

"""
i = 17

img = images[i]
laplace_align, max_blur  = laplace_threshold(img, thresh=20)
blurred_align = mask_blur(laplace_align, j=101)
for n in range(50):
        blurred_align = mask_blur(blurred_align, n)
print(i)
print('a')
cv.imshow('window', img)
cv.waitKey(5000)
print('b', np.max(laplace_align))
cv.imshow('window', laplace_align)
cv.waitKey(5000)
print('c')
cv.imshow('window', blurred_align)
cv.waitKey(5000)
