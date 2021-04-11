import cv2 as cv
import numpy as np

src = cv.imread('/Users/anthonyesposito/Pictures/Birbs/ExportDSCF0110.jpg')
src2 = cv.imread('/Users/anthonyesposito/Pictures/Birbs/ExportDSCF0875.jpg')
cv.imshow('a window', src)
cv.waitKey(5000)
cv.imshow('another window', src2)
cv.waitKey(5000)