from sklearn.decomposition import PCA
import sys
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os
import time

def import_describe(directory, hist_thresh):
    a = os.walk(directory)
    for root, dirs, files in a:
        b = files
    
    file_nums = b.copy()

    for i in range(len(b)):
        a = filter(str.isdigit, b[i])
        file_nums[i] = int(''.join(a))

        print(i,file_nums[i])

    image1 = cv.imread(directory+b[0], cv.IMREAD_COLOR)
    
    images = np.zeros((len(b), image1.shape[0], image1.shape[1], image1.shape[2]), 'uint8')
    mask = np.zeros((len(b), image1.shape[0], image1.shape[1]), 'uint8')
    hist_width = 256-hist_thresh
    histograms = np.zeros((len(b), (hist_width)*3))

    for i in range(0, len(b)):
        images[i] = cv.imread(directory+b[i], cv.IMREAD_COLOR)
        
        for j in range(images[i].shape[2]):
            hist, bin_edges = np.histogram(images[i,:,:,j], bins=np.arange(257))
            histograms[i,(j*hist_width):(j*hist_width+hist_width)] = hist[hist_thresh:256]
        
    return b, images, file_nums, mask, histograms

def image_sort(images, filenames, file_nums, histograms, n_comps, color_channels, hist_thresh):

    images_pca = np.zeros((len(filenames), color_channels*n_comps))
    pca = PCA(n_components=n_comps)

    colors = ['b', 'g', 'r']
    markers = ['.', 'x', '+']

    hist_width = 256-hist_thresh

    for i in range(color_channels):
        plt.figure(num=(colors[i]))
        pca.fit(histograms[:,(i*hist_width):(i*hist_width+hist_width)])
        pca_temp = pca.transform(histograms[:,(i*hist_width):(i*hist_width+hist_width)])
        for j in range(n_comps):
            pca_temp_ij = pca_temp[...,j]
            pca_temp_ij_min = pca_temp_ij.min()
            pca_temp_ij_max = pca_temp_ij.max()
            images_pca[...,j*color_channels+i] = ( pca_temp_ij - pca_temp_ij_min ) / ( ( j + 1 ) *  pca_temp_ij_max )
            plt.plot(file_nums, images_pca[:,j*color_channels+i], markers[j], color=colors[i])
        plt.show(block=False)
        plt.pause(5)

    image_corr = np.corrcoef(images_pca)
    print(np.argmax(np.sum(image_corr, axis=0)))
    image_corr_argsort = np.argsort(image_corr, axis=1)
    image_corr_maxmin = np.zeros(image_corr_argsort.shape, dtype='uint8')
    for i in range(image_corr_maxmin.shape[0]):
        for j in range(image_corr_maxmin.shape[1]):
            image_corr_maxmin[i,j] = list(image_corr_argsort[i]).index(j)
            print(i, '  ', file_nums[i], '  ', j, '  ',image_corr_maxmin[i,j], '  ', file_nums[j])
    print(image_corr_argsort)
    print(image_corr_maxmin)

    return image_corr_argsort, image_corr_maxmin

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

def registration(im1,im2, warp_mode=cv.MOTION_EUCLIDEAN, number_of_iterations=1000, termination_eps=1e-3):
    # Read the images to be aligned

    # Convert images to grayscale
    if len(im1.shape) == 3:
        im1_gray = cv.cvtColor(im1,cv.COLOR_BGR2GRAY)
        im2_gray = cv.cvtColor(im2,cv.COLOR_BGR2GRAY)
    else:
        im1_gray = im1
        im2_gray = im2
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
    

    # Specify the threshold of the increment
    # in the correlation coefficient between two iterations
    

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
    for i in range(20):
        ind = (ind+1) % 2

        if ind == 0:
            cv.imshow("Image", im1)
            cv.waitKey(10)
        else:
            cv.imshow("Image", im2_aligned)
            cv.waitKey(10)

        

