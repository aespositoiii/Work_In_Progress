"""
@file laplace_demo.py
@brief Sample code showing how to detect edges using the Laplace operator
"""
import sys
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import time

def blurred_laplace(src):
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
    ret,abs_dst = cv.threshold(abs_dst, 15,255,cv.THRESH_TOZERO)
    abs_dst_align = abs_dst

    # [display]
    '''ind=0
    while True:
        ind = (ind + 1) % 256
        n = np.median(abs_dst[abs_dst>10])+np.std(abs_dst[abs_dst>10])
        ret,abs_dst_disp = cv.threshold(abs_dst, ind,255,cv.THRESH_TOZERO)
        abs_dst_disp = abs_dst_disp**5
        cv.imshow('window_name', abs_dst_disp)
        print(ind)
        c = cv.waitKey(1000)
        if c == 27:
            break'''

    # [display]
    
    #n = np.median(abs_dst[abs_dst>10])+np.std(abs_dst[abs_dst>10])
    cv.imshow('window', abs_dst)
    
    c = cv.waitKey(5000)

    blur_size = (101,101)

    for j in range(10):
        '''if j==0:
            blur_size = (3,3)
        elif j%2==0:
            blur_size = (j+1,j+1)
        else:
            blur_size = (j,j)'''
        blur_size = (101, 101)
        abs_dst = cv.GaussianBlur(abs_dst, blur_size,0)
        peak_new = np.max(abs_dst)
        abs_dst = ( abs_dst ) / peak_new
        print(j,peak_new)
        cv.imshow('window', abs_dst)
        cv.waitKey(10)
        
    
    
    return abs_dst

imageName = '/Users/anthonyesposito/Pictures/macroni/Rosasite_w_Conacalcite/1/JPG/ExportDSCF69382022-42-50.jpg'

img = cv.imread(imageName, cv.IMREAD_COLOR) # Load an Image

laplace_align, blurred  = blurred_laplace(img)

print('d')
cv.imshow('window', img)
cv.waitKey(1000)
print('e')
cv.imshow('window', blurred)
cv.waitKey(1000)



