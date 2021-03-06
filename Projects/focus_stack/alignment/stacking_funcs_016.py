from sklearn.decomposition import PCA
import sys
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import glob

def import_describe(directory, hist_thresh):
    b = glob.glob(directory+ '*.jpg')
    file_nums = b.copy()

    for i in range(len(b)):
        a = filter(str.isdigit, b[i])
        file_nums[i] = int(''.join(a))

    image1 = cv.imread(b[0], cv.IMREAD_COLOR)
    
    images = np.zeros((len(b), image1.shape[0], image1.shape[1], image1.shape[2]), 'uint8')
    mask = np.zeros((len(b), image1.shape[0], image1.shape[1]), 'uint8')
    hist_width = 256-hist_thresh
    histograms = np.zeros((len(b), (hist_width)*3))

    for i in range(0, len(b)):
        images[i] = cv.imread(b[i], cv.IMREAD_COLOR)
        
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
        plt.pause(1)

    image_corr = np.corrcoef(images_pca)
    
    image_corr_argsort = np.argsort(image_corr, axis=1)
    image_corr_maxmin = np.zeros(image_corr_argsort.shape, dtype='uint8')
    image_corr_sort = np.zeros(image_corr_argsort.shape, dtype='uint8')
    image_count = images.shape[0]
    for i in range(image_count):
        for j in range(image_count):
            image_corr_maxmin[i,j] = list(image_corr_argsort[i]).index(j)
        for j in range(image_count):
            image_corr_sort[i,j] = list(image_corr_maxmin[i]).index(j)
    
    start_image = np.argmax(np.sum(np.sort(image_corr, axis=0)[...,(image_count-6):], axis=0))

    order = [start_image]
    image_list = list(range(image_count))
    image_list.remove(start_image)
    n=1

    while len(image_list) > 0:
        order.reverse()
        ref_array = np.ravel(image_corr_sort[order,...], order='F')
        order.reverse()
        for i in range(ref_array.shape[0]-1, -1, -1):
        
            if ref_array[i] not in order:
                order.append(ref_array[i])
                image_list.remove(int(ref_array[i]))
                break
        

    for i in range(len(order)):
        print(file_nums[order[i]])

    return order




def laplace_threshold(src, thresh, norm_blur):
    
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

    abs_dst_check = np.max(cv.GaussianBlur(abs_dst, (norm_blur,norm_blur),0))

    return abs_dst_align, abs_dst_check




def mask_blur(img, thresh=15, norm_blur=11, n_iter=50):

    abs_dst, norm  = laplace_threshold(img, thresh, norm_blur)
    
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
        im1_gray = np.copy(im1)
        im2_gray = np.copy(im2)
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

    return warp_matrix




def img_warp(im2, warp_matrix, warp_mode):

    sz = im2.shape

    if warp_mode == cv.MOTION_HOMOGRAPHY :
        # Use warpPerspective for Homography 
        im2_aligned = cv.warpPerspective (im2, warp_matrix, (sz[1],sz[0]), flags=cv.INTER_LINEAR + cv.WARP_INVERSE_MAP)
    else :
        # Use warpAffine for Translation, Euclidean and Affine
        im2_aligned = cv.warpAffine(im2, warp_matrix, (sz[1],sz[0]), flags=cv.INTER_LINEAR + cv.WARP_INVERSE_MAP);

    return im2_aligned

#def combine(im1, im2, im1_mask, im2_mask):

def reg_comb(images, order, thresh=15, norm_blur=11, n_iter=50, exp=2, warp_mode=cv.MOTION_EUCLIDEAN, number_of_iterations=1000, termination_eps=1e-3):

    comb = images[order[0]]
    comb_grad = mask_blur(comb, thresh, norm_blur, n_iter)+1  

    for i in order[1:]:
        t1 = time.time()

        img = images[i]
        img_grad = mask_blur(img=img, thresh=thresh, n_iter=n_iter)

        warp_matrix = registration(comb, img, warp_mode, number_of_iterations, termination_eps)

        img_warped = img_warp(img, warp_matrix, warp_mode)
        img_grad_warped = img_warp(img_grad, warp_matrix, warp_mode)+1

        comb_grad_float = comb_grad.astype('float32') ** exp
        img_grad_warped_float = img_grad_warped.astype('float32') ** exp

        comb_mask = comb_grad_float / (comb_grad_float+img_grad_warped_float)
        img_mask_warped = img_grad_warped_float / ( comb_grad_float+img_grad_warped_float)

        comb_temp = np.copy(comb)

        for i in range(comb.shape[2]):
            comb_temp[:,:,i] = comb[:,:,i] * comb_mask + img_warped[:,:,i] * img_mask_warped

        comb_grad = np.maximum(comb_grad, img_grad_warped)

        comb = np.copy(comb_temp)

        print(time.time()-t1)

        cv.imshow('window', comb)
        cv.waitKey(1)

    return comb