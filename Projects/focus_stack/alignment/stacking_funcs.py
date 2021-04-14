import sys
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os
import time

def get_images(directory):
    a = os.walk(directory)
    for root, dirs, files in a:
        b = files
    
    images = []
    
    for i in range(len(b)):
        images.append(cv.imread(directory+b[i], cv.IMREAD_COLOR))
        
    
    return images

def laplace_threshold(src, thresh=15):
    # [variables]
    # Declare the variables we are going to use
    t1 = time.time()
    ddepth = cv.CV_16S
    kernel_size = 3

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

    ret,abs_dst = cv.threshold(abs_dst, thresh,255,cv.THRESH_TOZERO)
    abs_dst_align = abs_dst

    abs_dst_check = np.max(cv.GaussianBlur(abs_dst, (21,21),0))

    return abs_dst_align, abs_dst_check
    
def mask_blur(abs_dst, j=3):

    if j==0:
        blur_size = (3,3)
    elif j%2==0:
        blur_size = (j+1,j+1)
    else:
        blur_size = (j,j)
    
    abs_dst = cv.GaussianBlur(abs_dst, blur_size,0)
    peak_new = np.max(abs_dst)
    abs_dst = ( abs_dst ) / peak_new 
    
    return abs_dst
"""
imageName = '/Users/anthonyesposito/Pictures/macroni/Rosasite_w_Conacalcite/1/JPG/ExportDSCF69382022-42-50.jpg'

img = cv.imread(imageName, cv.IMREAD_COLOR) # Load an Image

laplace_align  = laplace_threshold(img, thresh=15)
blurred_align = mask_blur(laplace_align, j=1)

for i in range(2,50):
    blurred_align = mask_blur(blurred_align, i)

print('a')
cv.imshow('window', img)
cv.waitKey(5000)
print('c')
cv.imshow('window', laplace_align)
cv.waitKey(5000)
print('c')
cv.imshow('window', blurred_align)
cv.waitKey(5000)
"""