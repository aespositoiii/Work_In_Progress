from sklearn.decomposition import PCA
import sys
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import glob
import shutil
import math
import PIL

# import_describe handles importing of images into memory.

class Stacking:

    def import_describe(directory, ext, hist_min=0, hist_max=255):

        # retrieve the list of filenames for images in the directory.

        b = glob.glob(directory + '*' + ext)         

        '''
        If a folder of transformed images already exists, get the list of images in the transformed directory.
        
        If the number of images is double the number of images in the slice images import, the transformed
        images and masks will be imported.  Otherwise the folder will be destroyed and the 
        directory and transformed images will be removed.

        + Instead of importing the images in a random order, export the list of sorted filenames at the image export step.
        '''
        file_nums = np.zeros(len(b))

        image1 = cv.imread(b[0], cv.IMREAD_COLOR)
        images = np.zeros((len(b), image1.shape[0], image1.shape[1], image1.shape[2]), 'uint8')
        
        hist_width = hist_max - hist_min
        histograms = np.zeros((len(b), (hist_width)*3))
        
        # Read in the

        for i in range(len(b)):
            a = ''
            for j in range(len(b[i])):
                if ( b[i][-j].isdigit() == False ) & ( len(a) == 0 ):
                    continue
                elif ( b[i][-j].isdigit() == True ):
                    a += b[i][-j]
                else:
                    break

            a = list(a)
            a.reverse()
            a = float((''.join(a)))
            file_nums[i] = a

            t1 = time.time()
            try:
                images[i] = cv.imread(b[i], cv.IMREAD_COLOR)
            except:
                image1 = cv.imread(b[0], cv.IMREAD_COLOR)
                images = np.zeros((len(b), image1.shape[0], image1.shape[1], image1.shape[2]), 'uint8')
                images[i] = image1

            #cv.imshow('window', images[i])
            #cv.waitKey(500)
            for j in range(images[i].shape[2]):
                hist, bin_edges = np.histogram(images[i,:,:,j], bins=np.arange(257))
                histograms[i,(j*hist_width):(j*hist_width+hist_width)] = hist[hist_min:hist_max]
            print(i, '  ', time.time()-t1)

            
            file_nums = file_nums.astype('uint32')

            

            
                
            
        return b, images, file_nums, histograms




    def image_sort(images, filenames, file_nums, histograms, n_comps, color_channels, hist_min=0, hist_max=256):

        images_pca = np.zeros((len(filenames), color_channels*n_comps))
        pca = PCA(n_components=n_comps)

        colors = ['b', 'g', 'r']
        markers = ['.', 'x', '+']

        hist_width = hist_max - hist_min

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
        trans_on = [None]

        image_list = list(range(image_count))
        image_list.remove(start_image)

        n = np.ones(image_count, dtype='i')

        while len(image_list) > 0:

            next_check = np.where(n[order] == n[order].min())[0][0]
            
            check_item = image_corr_sort[order[next_check]][ image_count - n[order[next_check]] - 1 ]

            if check_item not in order:
                order.append(check_item)
                trans_on.append(next_check)
                print(len(order), '  ', file_nums[order[next_check]], '  ', file_nums[check_item])
                image_list.remove(check_item)

            n[order[next_check]] += 1

            '''order.reverse()
            ref_array = np.ravel(image_corr_sort[order,...], order='F')
            order.reverse()
            for i in range(ref_array.shape[0]-1, -1, -1):
            
                if ref_array[i] not in order:
                    order.append(ref_array[i])
                    image_list.remove(int(ref_array[i]))
                    break'''
            

        return order, trans_on



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
        warp_mode = warp_mode

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


    def reg_comb(images, order, trans_on, file_nums, mask, warp_mode=cv.MOTION_EUCLIDEAN, number_of_iterations=1000, termination_eps=1e-3):
        t0 = time.time()

        for i in range(len(order)):
            t1 = time.time()
            if i != 0:
                warp_matrix = Stacking.registration(images[order[trans_on[i]]], images[order[i]], warp_mode, number_of_iterations, termination_eps)
                images[order[i]] = Stacking.img_warp(images[order[i]], warp_matrix, warp_mode)
                mask[order[i]] = Stacking.img_warp(mask[order[i]], warp_matrix, warp_mode)
                print(i, '  ', file_nums[order[i]], '  ', file_nums[order[trans_on[i]]], '  ', time.time()-t1, 'sec   ', time.time()-t0, 'sec')

        t2 = time.time()

        mask_select = np.zeros(mask.shape, dtype-'uint8')
        mask_select[mask == np.max(mask, axis=0)] = 1
        mask = ( mask * mask_select ) / np.sum(mask_select)

        comb = np.zeros(images.shape[1:])


        for i in range(images.shape[3]):
            images[:,:,:,i] = images[:,:,:,i] * mask
            comb[:,:,i] = np.sum(images[:,:,:,i], axis=0)

        cv.imshow('comb', comb)
        cv.waitKey(1)
        cv.destroyAllWindows()
        return comb