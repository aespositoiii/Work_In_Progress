import cv2 as cv
import os

directory = '/Users/anthonyesposito/Pictures/macroni/Rosasite_w_Conacalcite/1/JPG/'
a = os.walk(directory)
for root, dirs, files in a:
    b = files
print(len(b))
images = []
for i in range(len(b)):
    images.append(cv.imread(directory+b[i], cv.IMREAD_COLOR))
    cv.imshow('window', images[i])
    cv.waitKey(100)
print(len(images))