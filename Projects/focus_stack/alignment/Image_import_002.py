import cv2 as cv
import numpy as np
import time
import os

directory = '/Users/anthonyesposito/Pictures/macroni/Rosasite_w_Conacalcite/1/JPG/'
a = os.walk(directory)
for root, dirs, files in a:
    b = files

images = list([])
for i in range(len(b)):
    images.append(cv.imread(directory+b[i], cv.IMREAD_COLOR))
    print(directory + b[i])

for i in images:
    cv.imshow('window', i)
    cv.waitKey(1000)