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
    
    image1 = cv.imread(directory+b[0], cv.IMREAD_COLOR)
    
    images = np.zeros((len(b), image1.shape[0], image1.shape[1], image1.shape[2]), 'uint8')
    align = np.zeros((len(b), image1.shape[0], image1.shape[1]), 'uint8')
    mask = np.zeros((len(b), image1.shape[0], image1.shape[1]), 'uint8')

    for i in range(1, len(b)):
        images[i] = cv.imread(directory+b[i], cv.IMREAD_COLOR)
        
    
    return images, align, mask

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

    abs_dst_check = np.max(cv.GaussianBlur(abs_dst, (11,11),0))

    return abs_dst_align, abs_dst_check
    
def mask_blur(abs_dst, norm, n_iter=50):

    for i in range(3, ( 2 * n_iter), 2):

        abs_dst = cv.GaussianBlur(abs_dst, (i,i), 0)
        abs_dst = cv.normalize(abs_dst, abs_dst, 0, norm, cv.NORM_MINMAX)
        
    return abs_dst
