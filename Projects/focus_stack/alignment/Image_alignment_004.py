import cv2 as cv
import numpy as np
import time

def registration(im1,im2, warp_mode=cv.MOTION_EUCLIDEAN, number_of_iterations=1000, termination_eps=1e-3)
    # Read the images to be aligned
    im1 =  cv.imread("/Users/anthonyesposito/Pictures/macroni/Rosasite_w_Conacalcite/1/JPG/ExportDSCF69412022-43-07.jpg");
    im2 =  cv.imread("/Users/anthonyesposito/Pictures/macroni/Rosasite_w_Conacalcite/1/JPG/ExportDSCF69422022-43-14.jpg");

    # Convert images to grayscale
    im1_gray = cv.cvtColor(im1,cv.COLOR_BGR2GRAY)
    im2_gray = cv.cvtColor(im2,cv.COLOR_BGR2GRAY)

    # Find size of image1
    sz = im1.shape

    # Define the motion model
    warp_mode = cv.MOTION_EUCLIDEAN

    # Define 2x3 or 3x3 matrices and initialize the matrix to identity
    if warp_mode == cv.MOTION_HOMOGRAPHY :
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else :
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Specify the number of iterations.
    number_of_iterations = 1000;

    # Specify the threshold of the increment
    # in the correlation coefficient between two iterations
    termination_eps = 1e-3;

    # Define termination criteria
    criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)

    t1 = time.time()

    # Run the ECC algorithm. The results are stored in warp_matrix.
    cc, warp_matrix = cv.findTransformECC(im1_gray,im2_gray,warp_matrix, warp_mode, criteria, None, 1)

    if warp_mode == cv.MOTION_HOMOGRAPHY :
        # Use warpPerspective for Homography 
        im2_aligned = cv.warpPerspective (im2, warp_matrix, (sz[1],sz[0]), flags=cv.INTER_LINEAR + cv.WARP_INVERSE_MAP)
    else :
        # Use warpAffine for Translation, Euclidean and Affine
        im2_aligned = cv.warpAffine(im2, warp_matrix, (sz[1],sz[0]), flags=cv.INTER_LINEAR + cv.WARP_INVERSE_MAP);

    # Show final results
    ind = 0

    '''for i in range(100):
        ind = (ind+1) % 2

        if ind == 0:
            cv.imshow("Image", im1[1500:1600,1500:1600])
            cv.waitKey(100)
        else:
            cv.imshow("Image", im2_aligned[1500:1600,1500:1600] )
            cv.waitKey(100)'''
    print(time.time() - t1)

    for i in range(100):
        ind = (ind+1) % 2

        if ind == 0:
            cv.imshow("Image", im1)
            cv.waitKey(10)
        else:
            cv.imshow("Image", im2_aligned)
            cv.waitKey(10)

    difference = im1-im2_aligned
    print(np.var(difference))


im1 =  cv.imread("/Users/anthonyesposito/Pictures/macroni/Rosasite_w_Conacalcite/1/JPG/ExportDSCF69412022-43-07.jpg");
im2 =  cv.imread("/Users/anthonyesposito/Pictures/macroni/Rosasite_w_Conacalcite/1/JPG/ExportDSCF69422022-43-14.jpg");