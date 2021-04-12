"""
@file laplace_demo.py
@brief Sample code showing how to detect edges using the Laplace operator
"""
import sys
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

def main(argv):
    # [variables]
    # Declare the variables we are going to use
    ddepth = cv.CV_16S
    kernel_size = 3
    window_name = "Laplace Demo"
    # [variables]

    # [load]
    imageName = argv[0] if len(argv) > 0 else '/Users/anthonyesposito/Pictures/macroni/Rosasite_w_Conacalcite/1/JPG/ExportDSCF69422022-43-14.jpg'

    src = cv.imread(cv.samples.findFile(imageName), cv.IMREAD_COLOR) # Load an image

    # Check if image is loaded fine
    if src is None:
        print ('Error opening image')
        print ('Program Arguments: [image_name -- default lena.jpg]')
        return -1
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
    cv.namedWindow(window_name, cv.WINDOW_AUTOSIZE)

    # [laplacian]
    # Apply Laplace function
    dst = cv.Laplacian(src_gray, ddepth, ksize=kernel_size)
    #dst = dst**2
    # [laplacian]

    # [convert]
    # converting back to uint8
    abs_dst = cv.convertScaleAbs(dst)
    ret,abs_dst_disp = cv.threshold(abs_dst,np.median(abs_dst[abs_dst>10])+np.std(abs_dst[abs_dst>10]),255,cv.THRESH_TOZERO)
    #plt.hist(abs_dst[abs_dst>0].ravel(),256,[0,256]); plt.show()
    # [convert]
    print(abs_dst.max())
    print(abs_dst.min())
    print(np.unique(abs_dst).shape)

    # [display]
    cv.imshow(window_name, abs_dst)
    print(src.shape, '\n\n', abs_dst.shape)
    cv.waitKey(0)
    # [display]
    ind =0
    
    while True:
        ind = (ind + 1) % 256
        #n = np.median(abs_dst[abs_dst>10])+np.std(abs_dst[abs_dst>10])
        ret,abs_dst_disp = cv.threshold(abs_dst, ind,255,cv.THRESH_TOZERO)
        cv.imshow('window_name', abs_dst_disp)
        print(ind)
        c = cv.waitKey(10)
        if c == 27:
            break

    return 0

if __name__ == "__main__":
    main(sys.argv[1:])
