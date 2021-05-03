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

def import_describe(directory, ext, hist_min=0, hist_max=255):

    b = glob.glob(directory + '*' + ext) 

    if os.path.exists(directory + 'transform/'):

        c = glob.glob(directory + 'transform/' + '*' + ext)

        if len(c) == len(b)*2:
            transformed = True
            histograms = None
            image1 = cv.imread(b[0], cv.IMREAD_COLOR)
            images = np.zeros((len(b), image1.shape[0], image1.shape[1], image1.shape[2]), 'uint8')
            lap_mask = np.zeros((len(b), image1.shape[0], image1.shape[1]), 'uint8')
            file_nums = np.zeros(len(b))

            for i in range(len(c)):
                a = ''
                im = False
                mask = False
                for j in range(len(c[i])):
                    cij = c[i][-j]
                    if (cij.isdigit() == False) & (len(a) == 0):
                        continue
                    elif cij.isdigit() == True:
                        a = a + cij
                        if c[i][-j+1] == 'i':
                            im = True
                        elif c[i][-j+1] == 'm':
                            mask = True
                    else:
                        break
                
                a = list(a)
                a.reverse()
                a = int((''.join(a)))
                if mask == True:
                    lap_mask[a] = cv.imread(c[i], cv.IMREAD_GRAYSCALE)
                elif im == True:
                    file_nums[a] = a
                    images[a] = cv.imread(c[i], cv.IMREAD_COLOR)
                
                print(im, a)

        else:
            shutil.rmtree(directory + 'transform/', ignore_errors=True)

        t0 = time.time()
        '''for i in range(images.shape[0]):
                        print(i)
                        cv.imshow('images', images[i])
                        cv.waitKey(1)
                        cv.imshow('mask', lap_mask[i])
                        cv.waitKey(1000)'''
    
    if not os.path.exists(directory + 'transform/'):
        transformed = False
        file_nums = np.zeros(len(b))

        for i in range(len(b)):
            a = ''
            for j in range(len(b[i])):
                if ( b[i][-j].isdigit() == False ) & ( len(a) == 0 ):
                    continue
                elif ( b[i][-j].isdigit() == True ):
                    print(b[i][-j], type((b[i][-j])))
                    a += b[i][-j]
                else:
                    break

            a = list(a)
            a.reverse()
            a = float((''.join(a)))
            file_nums[i] = a
        
        file_nums = file_nums.astype('uint32')

        for i in range(file_nums.shape[0]):
            print(i, ' ', file_nums[i])

        image1 = cv.imread(b[0], cv.IMREAD_COLOR)
        
        images = np.zeros((len(b), image1.shape[0], image1.shape[1], image1.shape[2]), 'uint8')
        lap_mask = None
        hist_width = hist_max - hist_min
        histograms = np.zeros((len(b), (hist_width)*3))

        for i in range(0, len(b)):
            t1 = time.time()
            images[i] = cv.imread(b[i], cv.IMREAD_COLOR)
            #cv.imshow('window', images[i])
            #cv.waitKey(500)
            for j in range(images[i].shape[2]):
                hist, bin_edges = np.histogram(images[i,:,:,j], bins=np.arange(257))
                histograms[i,(j*hist_width):(j*hist_width+hist_width)] = hist[hist_min:hist_max]
            print(i, '  ', time.time()-t1)
        
    return b, images, file_nums, lap_mask, histograms, transformed




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
        
    return abs_dst, norm




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


def reg_comb(images, order, trans_on, file_nums, transformed, ext=None, directory=None, lap_mask=None, thresh=15, norm_blur=11, n_iter=50, exp=2, warp_mode=cv.MOTION_EUCLIDEAN, number_of_iterations=1000, termination_eps=1e-3):
    t0 = time.time()
    
    
    if transformed == False:
        lap_mask = np.zeros(images.shape[0:3], dtype='uint8')
        norm = np.zeros(len(order), dtype='uint8')

        for i in range(len(order)):
            t1 = time.time()
            lap_mask[order[i]], norm[order[i]] = mask_blur(img=images[order[i]], thresh=thresh, n_iter=n_iter)
            if i != 0:
                warp_matrix = registration(images[order[trans_on[i]]], images[order[i]], warp_mode, number_of_iterations, termination_eps)
                images[order[i]] = img_warp(images[order[i]], warp_matrix, warp_mode)
                lap_mask[order[i]] = img_warp(lap_mask[order[i]], warp_matrix, warp_mode)
                print(i, '  ', file_nums[order[i]], '  ', file_nums[order[trans_on[i]]], '  ', time.time()-t1, 'sec   ', time.time()-t0, 'sec')

        directory_transform = directory + 'transform/'  
        os.mkdir(directory_transform)
        for i in range(len(order)):            
            cv.imwrite((directory_transform + str(i) + 'image' + ext), images[i])
            cv.imwrite((directory_transform + str(i) + 'mask' + ext), lap_mask[i])

    t2 = time.time()


    lap_mask_float = np.zeros(lap_mask.shape)
    lap_mask_float = np.float_power(lap_mask, exp)
    lap_mask_sum = lap_mask_float.sum(axis=0)

    comb = np.zeros(images[0].shape, dtype='uint8')
    comb_mask = np.zeros(lap_mask_float.shape)

    lap_mask_norm = np.zeros(lap_mask_float.shape)

    
    t3 = time.time()
    print(t3-t2, '   ', t3-t0)

    iter = 20
    last_iter = False
    
    for n in range(iter) :
        
        lap_mask_sum = lap_mask_float.sum(axis=0)
        
        for i in range(lap_mask.shape[0]):
            t4 = time.time()
            lap_mask_norm[i][lap_mask_sum>0] = lap_mask_float[i][lap_mask_sum>0] / lap_mask_sum[lap_mask_sum>0]
            t5 = time.time()
            print(i, '  ', t5-t4, '   ', t5-t0)
    
        lap_mask_norm[comb_mask>(0)] = 0
        lap_mask_norm_sum = lap_mask_norm.sum(axis=0)
        cv.imshow('Lap Mask Norm', lap_mask_norm_sum)
        cv.waitKey(1)

        '''for i in range(lap_mask_norm.shape[0]):
            cv.imshow('window', lap_mask_norm[i])
            cv.waitKey(1000)    '''    



        for i in range(images.shape[3]):
            t6 = time.time()
            comb[:,:,i][lap_mask_norm_sum>0] = comb[:,:,i][lap_mask_norm_sum>0] + (images[:,:,:,i][:,lap_mask_norm_sum>0] * lap_mask_norm[:,lap_mask_norm_sum>0]).sum(axis=0)
            t7 = time.time()
            print(i, '   ', t7-t6, '   ', t7-t0)
        
        cv.imshow('comb', comb)
        cv.waitKey(1)

        if (n == iter-1) | (last_iter==True):
            break
        
        try:
            print('Press [ctrl-c] for last iteration')
            ti = time.time()
            tb = 11
            while (time.time() - ti) < 10:
                pass
            print('Continuing to next iteration')
        except KeyboardInterrupt:
            last_iter = True
            print('\n\nInitiating last iteration...')
        
        

        comb_mask = np.ones(comb_mask.shape)
        comb_mask = lap_mask_sum * comb_mask
        cv.imshow('comb mask', comb_mask[0])
        cv.waitKey(1)

        for i in range(lap_mask_float.shape[0]):
            t8 = time.time()
            lap_mask_float[i] = cv.GaussianBlur(lap_mask_float[i], ((9 + 2 * n),(9 + 2 * n)), 0)
            if (n == iter-2) | (last_iter == True):
                lap_mask_float = lap_mask_float + 1
            t9 = time.time()
            print(i, '  ', t9-t8)
        print('n = ', n)

        


    '''comb = np.zeros(images[0].shape, dtype='uint8')
    comb_mask = np.zeros(lap_mask[0].shape, dtype='uint8') 
    diff_mask = np.zeros(lap_mask[0].shape, dtype='uint8')
    temp_maks = np.zeros(lap_mask[0].shape, dtype='uint8')

    for i in range(lap_mask.max()-10, 0, -10):
        for j in range(len(order)):
            t1 = time.time()
            img = np.copy(images[order[j]])
            ret, temp_mask = cv.threshold(lap_mask[order[j]], i, 1,cv.THRESH_BINARY)
            diff_mask = cv.bitwise_and(temp_mask, comb_mask, diff_mask, mask = np.ones(comb_mask.shape, dtype='uint8'))
            temp_mask = temp_mask - diff_mask
            for k in range(comb.shape[2]):
                img[:,:,k][temp_mask==0] = 0
                comb[:,:,k] = comb[:,:,k] + img[:,:,k]
            comb_mask = comb_mask + temp_mask
            print(i, '  ', j, '  ', time.time()-t1)
            cv.imshow('mask', comb_mask*255)
            cv.waitKey(1)
            cv.imshow('comb', comb)
            cv.waitKey(1)'''

    '''for i in order[1:]:
        t1 = time.time()

        img = images[i]
        img_grad = mask_blur(img=img, thresh=thresh, n_iter=n_iter)

        warp_matrix = registration(comb, img, warp_mode, number_of_iterations, termination_eps)

        img_warped = img_warp(img, warp_matrix, warp_mode)
        img_grad_warped = img_warp(img_grad, warp_matrix, warp_mode)

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
        cv.waitKey(1)'''
    cv.destroyAllWindows()
    return comb