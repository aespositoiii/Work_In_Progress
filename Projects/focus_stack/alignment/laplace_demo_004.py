"""
@file laplace_demo.py
@brief Sample code showing how to detect edges using the Laplace operator
"""
import sys
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import time

def blurred_laplace(src, blur_size=(201,201)):
    # [variables]
    # Declare the variables we are going to use
    t1 = time.time()
    ddepth = cv.CV_16S
    kernel_size = 3
    window_name = "Laplace Demo"
    # [variables]

    # [load]
    imageName = '/Users/anthonyesposito/Pictures/macroni/Rosasite_w_Conacalcite/1/JPG/ExportDSCF69422022-43-14.jpg'

    src = cv.imread(cv.samples.findFile(imageName), cv.IMREAD_COLOR) # Load an image

    # [load]

    # [reduce_noise]
    # Remove noise by blurring with a Gaussian filter
    src = cv.GaussianBlur(src, (3, 3), 0)
    # [reduce_noise]

    # [convert_to_gray]
    # Convert the image to grayscale
    src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    # [convert_to_gray]

    # Create Window

    # [laplacian]
    # Apply Laplace function
    dst = cv.Laplacian(src_gray, ddepth, ksize=kernel_size)
    #dst = dst**2
    # [laplacian]

    # [convert]
    # converting back to uint8
    abs_dst = cv.convertScaleAbs(dst)

    # [convert]

    abs_dst = (abs_dst/(np.percentile(abs_dst[abs_dst>0], 10)))**1.5
    abs_dst = (abs_dst/abs_dst.max())*255
    abs_dst = abs_dst.astype('uint8')

    # [display]



    # [display]
    ind = 0

    for i in range(15):
        abs_dst = cv.GaussianBlur(abs_dst, blur_size,0)
        abs_dst = (abs_dst/abs_dst.max())*255
        abs_dst = abs_dst.astype('uint8')

    return abs_dst

imageName = '/Users/anthonyesposito/Pictures/macroni/Rosasite_w_Conacalcite/1/JPG/ExportDSCF69422022-43-14.jpg'

img = cv.imread(imageName, cv.IMREAD_COLOR) # Load an Image

blurred  = blurred_laplace(img)

cv.imshow('window', img)
cv.waitKey(5000)
cv.imshow('window', blurred)
cv.waitKey(5000)



