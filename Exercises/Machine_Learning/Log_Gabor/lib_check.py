#from Log_Gabor2 import log_gabor2
from log_gabor.log_gabor import log_gabor
import numpy as np
import cv2 as cv
import sys

filepath = "/Users/anthonyesposito/Desktop/GitWork/Work_In_Progress/Exercises/Machine_Learning/Log_Gabor/IMG_0727.jpg"

img = cv.imread(filepath)
imgG = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('image', img)
cv.imshow('Result', img)
cv.imshow('Filter', img)

# img_F = np.fft.fft2(imgG)

ohm = 200
sig_LG=.01
thet = 0
sigthet = .5

result, LG = log_gabor(imgG, 1/ohm, sig_LG, thet, sigthet)
print((1/ohm), '  ', sig_LG, '  ', thet, '  ', sig_LG)
print(img.shape, result.shape, LG.shape)
cv.imshow('Result', result)
cv.imshow('Filter', LG)
cv.waitKey(0)