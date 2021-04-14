from stacking_funcs import get_images, laplace_threshold, mask_blur
import sys
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os
import time

directory = '/Users/anthonyesposito/Pictures/macroni/Rosasite_w_Conacalcite/1/JPG/'

images = get_images(directory)

for i in range(len(images)):
    cv.imshow('window', images[i])
    cv.waitKey(100)