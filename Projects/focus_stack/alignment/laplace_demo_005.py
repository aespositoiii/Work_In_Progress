"""
@file laplace_demo.py
@brief Sample code showing how to detect edges using the Laplace operator
"""
import sys
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import time

def blurred_laplace(src, blur_size=(101,101)):
    # [variables]
    # Declare the variables we are going to use
    t1 = time.time()
    ddepth = cv.CV_16S
    kernel_size = 3
    window_name = "Laplace Demo"

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

    #plt.hist(abs_dst.ravel(),256,[0,256]); plt.yscale('log'); plt.show()
    # [convert]
    print('a')
    cv.imshow('window', abs_dst)
    cv.waitKey(5000)
    ret,abs_dst = cv.threshold(abs_dst, 10,255,cv.THRESH_TOZERO)
    abs_dst = abs_dst.astype('uint8')

    # [display]



    # [display]
    
    #n = np.median(abs_dst[abs_dst>10])+np.std(abs_dst[abs_dst>10])
    
    abs_dst = abs_dst**5
    print('b')
    cv.imshow('window_name', abs_dst)
    
    c = cv.waitKey(1000)
    
    for j in range(15):
        abs_dst = cv.GaussianBlur(abs_dst, blur_size,0)
        abs_dst = abs_dst.astype('uint8')
    print('c')
    cv.imshow('window_name', abs_dst)
    cv.waitKey(1000)
    
    return abs_dst

imageName = '/Users/anthonyesposito/Pictures/macroni/Rosasite_w_Conacalcite/1/JPG/ExportDSCF69402022-43-01.jpg'

img = cv.imread(imageName, cv.IMREAD_COLOR) # Load an Image

blurred  = blurred_laplace(img)

print('d')
cv.imshow('window', img)
cv.waitKey(5000)
print('e')
cv.imshow('window', blurred)
cv.waitKey(5000)



