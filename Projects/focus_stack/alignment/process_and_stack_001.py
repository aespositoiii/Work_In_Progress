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
from datetime import datetime
from skimage.transform import resize
from log_gabor.log_gabor import log_gabor
from aesop import aesop
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

            

            
                
            
        return b, images, histograms




    def image_sort(images, histograms, hist_min=0, hist_max=256):

        images_pca = np.zeros((images.shape[0], images.shape[3]**2))
        pca = PCA(n_components=images.shape[3])

        hist_width = hist_max - hist_min

        for i in range(images.shape[3]):
            
            pca.fit(histograms[:,(i*hist_width):(i*hist_width+hist_width)])
            pca_temp = pca.transform(histograms[:,(i*hist_width):(i*hist_width+hist_width)])
            for j in range(images.shape[3]):
                pca_temp_ij = pca_temp[...,j]
                pca_temp_ij_min = pca_temp_ij.min()
                pca_temp_ij_max = pca_temp_ij.max()
                images_pca[...,j*images.shape[3]+i] = ( pca_temp_ij - pca_temp_ij_min ) / ( ( j + 1 ) *  pca_temp_ij_max )

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
                image_list.remove(check_item)

            n[order[next_check]] += 1            

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


    def reg_comb(images, order, trans_on, mask, output_filename, warp_mode=cv.MOTION_EUCLIDEAN, number_of_iterations=1000, termination_eps=1e-3):
        t0 = time.time()

        for i in range(len(order)):
            t1 = time.time()
            if i != 0:
                warp_matrix = Stacking.registration(images[order[trans_on[i]]], images[order[i]], warp_mode, number_of_iterations, termination_eps)
                images[order[i]] = Stacking.img_warp(images[order[i]], warp_matrix, warp_mode)
                mask[order[i]] = Stacking.img_warp(mask[order[i]], warp_matrix, warp_mode)

        t2 = time.time()

        mask_select = np.zeros(mask.shape, dtype='uint8')
        mask_select[mask == np.max(mask, axis=0)] = 1
        mask = ( mask * mask_select ) / np.sum(mask_select)

        comb = np.zeros(images.shape[1:])


        for i in range(images.shape[3]):
            images[:,:,:,i] = images[:,:,:,i] * mask
            comb[:,:,i] = np.sum(images[:,:,:,i], axis=0)

        cv.imwrite(output_filename, comb)



    def batch_process(process_summary_filename, image_filenames, process_summary_dict,im_save, batch_stack):
        t1 = time.time()


        
        if batch_stack == 'batch':
            head, tail = os.path.split(process_summary_filename)

            now = datetime.now()
            batch_folder_name = head+'/Batch_{}{}{}_{}{}/'.format(now.year, now.month, now.day, now.hour, now.minute)
            print(batch_folder_name)
            os.mkdir(batch_folder_name)
            processed_image_folders = []
            for i in im_save:
                processed_image_folders.append(batch_folder_name + i + '/')
                os.mkdir(processed_image_folders[-1])

        

        for i in image_filenames:
            t2 = time.time()
            base_color = cv.imread(i, cv.IMREAD_COLOR)
            imgG = cv.cvtColor(base_color, cv.COLOR_BGR2GRAY)
            images = list([imgG])

            head, tail = os.path.split(i)
            filename = os.path.splitext(tail)
            
            for j in process_summary_dict:
                t3 = time.time()
                print(j['im_name'])
                
                im_name_string = 'Image Name: {}'.format(filename[0])
                im_prog_string = 'Image: {} of {}'.format(image_filenames.index(i)+1, len(image_filenames))
                process_name_string = 'Proocess Name: {}'.format(j['im_name'])
                procees_prog_string = 'Proocess: {} of {}'.format(process_summary_dict.index(j)+1, len(process_summary_dict))

                print('\n\n')
                print(im_name_string)
                print(im_prog_string)
                print(process_name_string)
                print(procees_prog_string)

                

                process = j['process']

                math_list = ['Add', 'Subtract', 'Multiply', 'Divide', 'Max', 'Min', '(im)^n']
                morph_list = ['Erode', 'Dilate', 'Open', 'Close', 'Morph_Gradient', 'Top_Hat', 'Black_Hat']

                if process == "Gabor":

                    kernel = cv.getGaborKernel((j['parameters'][0], j['parameters'][0]), j['parameters'][1], j['parameters'][2] * np.pi, j['parameters'][3] * np.pi, j['parameters'][4], j['parameters'][5] * np.pi, ktype=cv.CV_32F)

                    result = cv.filter2D(images[j['src_im_ind']], cv.CV_8U, kernel)
                    
                    


                elif process == "Log_Gabor":

                    result, LG = log_gabor(images[j['src_im_ind']], j['parameters'][0], j['parameters'][1], j['parameters'][2], j['parameters'][3])
                    histogram = np.histogram(result.ravel(), bins=256, range=[0,256])

                    if j['LG_Normalize'] == True:
                        peak = (np.where(histogram[0] == np.max(histogram[0])))[0][0]
                        if peak <= 127:
                            result[result < peak] = peak + (peak - result[result < peak])
                        elif peak > 127:
                            result[result > peak] = peak - (result[result > peak] - peak)
                            result = (255 * np.ones(result.shape, dtype='uint8')) - result                     

                    if j['truncate'] == True:
                        
                        zeroed = np.where([histogram[0] > 10**j['trunc_exp']])[1]

                        for k in zeroed:
                            result[result==k]=0

                        result[result>0] = 255

                    


                elif process == "Gauss":

                    result = cv.GaussianBlur(images[j['src_im_ind']], (j['parameters'][0], j['parameters'][0]), 0)
                    

                elif process == "Canny":

                    result = cv.Canny(images[j['src_im_ind']], j['parameters'][1], j['parameters'][0])
                                    

                elif process == "Laplace":

                    im = cv.GaussianBlur(images[j['src_im_ind']], (3, 3), 0)
                    result = cv.Laplacian(im, cv.CV_8U, ksize=j['parameters'][0])
                    
                    result = result.astype('uint8')


                elif process == "Aesop":
                    
                    temp_image = np.copy(images[j['src_im_ind']])

                    temp_image[temp_image > 0] = 1

                    if j['parameters'][3] == 0:
                        series_val = True
                    else:
                        series_val = False
                    
                    result = aesop.aesops_Filter(temp_image, kernel_size_start=j['parameters'][0], kernel_size_end=j['parameters'][1], kernel_step=j['parameters'][2], series=series_val, steps=j['parameters'][4])

                    result[result > 0] = 255

                
                ################################################################################################################################################                    
                
                elif process in math_list:
                    
                    if j['constant_term'] == 'N/A':
                        operand2 = images[j['src_im_ind2']]
                    
                    else:
                        operand2 = j['constant_term']

                    if process == 'Add':
                        result = cv.add(images[j['src_im_ind']], operand2)
                        
                    
                    elif process == 'Subtract':
                        result = cv.subtract(images[j['src_im_ind']], operand2)
                        

                    elif process == 'Multiply':
                        operand2 = operand2.astype('float32') / 255
                        images[j['src_im_ind']].astype('float32')
                        result = images[j['src_im_ind']] * operand2
                        result = result.astype('uint8')
                        #result = cv.multiply(images[j['src_im_ind']], operand2)
                        

                    elif process == 'Divide':
                        result = cv.divide(images[j['src_im_ind']], operand2)
                        

                    elif process == 'Max':
                        result = cv.max(images[j['src_im_ind']], operand2)
                        

                    elif process == 'Min':
                        result = cv.min(images[j['src_im_ind']], operand2)
                        

                    elif process == '(im)^n':
                        result = images[j['src_im_ind']].astype('float32')
                        result = result/255
                        result = cv.pow(result, operand2)
                        result = 255 * result
                        result = result.astype('uint8')
                        

                    elif process == 'e^(im)]':
                        result = images[j['src_im_ind']].astype('float32')
                        result = result/255
                        result = np.exp(images[j['src_im_ind']])
                        result = 255 * result
                        result = result.astype('uint8')                    
                        
                        
                    elif process == 'log(im)':
                        result = images[j['src_im_ind']].astype('float32')
                        result = result/255                    
                        result = np.log(images[j['src_im_ind']])
                        result = 255 * result
                        result = result.astype('uint8')                      
                        

                ############################################################################################################################################################                                            


                elif process == 'Binary':
                    ret, result = cv.threshold(images[j['src_im_ind']], j['parameters'][0], j['parameters'][1], cv.THRESH_BINARY)

                elif process == 'Inverse_Binary':
                    ret, result = cv.threshold(images[j['src_im_ind']], j['parameters'][0], j['parameters'][1], cv.THRESH_BINARY_INV)

                elif process == 'Truncated':
                    ret, result = cv.threshold(images[j['src_im_ind']], j['parameters'][0], j['parameters'][1], cv.THRESH_TRUNC)

                elif process == 'To_Zero':
                    ret, result = cv.threshold(images[j['src_im_ind']], j['parameters'][0], j['parameters'][1], cv.THRESH_TOZERO)                

                elif process == 'Inverse_To_Zero':
                    ret, result = cv.threshold(images[j['src_im_ind']], j['parameters'][0], j['parameters'][1], cv.THRESH_TOZERO_INV)

                elif process == 'Otsu_Bin':
                    ret, result = cv.threshold(images[j['src_im_ind']], j['parameters'][0], j['parameters'][1], cv.THRESH_BINARY+cv.THRESH_OTSU)                

                elif process == 'Triangle_Bin':
                    ret, result = cv.threshold(images[j['src_im_ind']], j['parameters'][0], j['parameters'][1], cv.THRESH_BINARY+cv.THRESH_TRIANGLE)

                elif process == 'Adaptive_Thresh_Mean_C':
                    result = cv.adaptiveThreshold(images[j['src_im_ind']], j['parameters'][0], cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, j['parameters'][1], j['parameters'][2])
                    
                elif process == 'Adaptive_Thresh_Gaussian_C':
                    result = cv.adaptiveThreshold(images[j['src_im_ind']], j['parameters'][0], cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, j['parameters'][1], j['parameters'][2])                

                elif process == 'Trunc_Hist':
                    result = np.copy(images[j['src_im_ind']])
                    histogram = np.histogram(result.ravel(), bins=256, range=[0,256])
                    zeroed = np.where([histogram[0] > 10**(j['parameters'][0])])[1]
                    for k in zeroed:
                        result[result==k]=0

                elif process == 'Norm_Inf':
                    result = cv.normalize(images[j['src_im_ind']], result, alpha=j['parameters'][0], beta=j['parameters'][1], norm_type=cv.NORM_INF, dtype=cv.CV_8U)

                elif process == 'Norm_L1':
                    j['parameters'][0] = float(j['parameters'][0])/255
                    j['parameters'][1] = float(j['parameters'][1])/255
                    
                    
                    images[j['src_im_ind']] = (images[j['src_im_ind']].astype('float32'))/255
                    
                    #result = cv.normalize(images[j['src_im_ind']], result, alpha=j['parameters'][0], beta=j['parameters'][1], norm_type=cv.NORM_L1, dtype=cv.CV_32F)
                    result = cv.normalize(images[j['src_im_ind']], result, alpha=1.0, beta=0.0, norm_type=cv.NORM_L1, dtype=cv.CV_32F)
                    result = result / result.max()
                    result = (result * 255).astype('uint8')
                    

                elif process == 'Norm_L2':
                    j['parameters'][0] = float(j['parameters'][0])/255
                    j['parameters'][1] = float(j['parameters'][1])/255
                    
                    
                    images[j['src_im_ind']] = (images[j['src_im_ind']].astype('float32'))/255
                    
                    #result = cv.normalize(images[j['src_im_ind']], result, alpha=j['parameters'][0], beta=j['parameters'][1], norm_type=cv.NORM_L2, dtype=cv.CV_8U)
                    result = cv.normalize(images[j['src_im_ind']], result, alpha=1.0, beta=0.0, norm_type=cv.NORM_L2, dtype=cv.CV_32F)
                    result = result / result.max()
                    result = (result * 255).astype('uint8')

                elif process == 'Norm_L2_Square':
                    j['parameters'][0] = float(j['parameters'][0])/255
                    j['parameters'][1] = float(j['parameters'][1])/255
                    
                    
                    images[j['src_im_ind']] = (images[j['src_im_ind']].astype('float32'))/255
                    
                    #result = cv.normalize(images[j['src_im_ind']], result, alpha=j['parameters'][0], beta=j['parameters'][1], norm_type=cv.NORM_L2SQR, dtype=cv.CV_8U)
                    result = cv.normalize(images[j['src_im_ind']], result, alpha=1.0, beta=0.0, norm_type=cv.NORM_L2SQR, dtype=cv.CV_32F)
                    result = result / result.max()
                    result = (result * 255).astype('uint8')

                elif process == 'Norm_Hamming':
                    result = cv.normalize(images[j['src_im_ind']], result, alpha=j['parameters'][0], beta=j['parameters'][1], norm_type=cv.NORM_HAMMING, dtype=cv.CV_8U)

                elif process == 'Norm_Hamming2':
                    result = cv.normalize(images[j['src_im_ind']], result, alpha=j['parameters'][0], beta=j['parameters'][1], norm_type=cv.NORM_HAMMING2, dtype=cv.CV_8U)

                elif process == 'Norm_Relative':
                    result = cv.normalize(images[j['src_im_ind']], result, alpha=j['parameters'][0], beta=j['parameters'][1], norm_type=cv.NORM_RELATIVE, dtype=cv.CV_8U)

                elif process == 'Norm_Min_Max':
                    result = cv.normalize(images[j['src_im_ind']], result, alpha=j['parameters'][0], beta=j['parameters'][1], norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)


                

                elif process == 'Invert':
                    result = (255 * np.ones(images[j['src_im_ind']].shape, dtype='uint8')) - images[j['src_im_ind']]

                elif process == 'Resize':
                    if j['Resize_Op'] == 'Reduce':
                        scale = j['parameters'][0]
                        scaled_size = (int(images[j['src_im_ind']].shape[0] * scale),int(images[j['src_im_ind']].shape[1] * j['parameters'][0]) )
                        result = resize(images[j['src_im_ind']], scaled_size, anti_aliasing=True)
                        result = (result*255).astype('uint8')

                    elif j['Resize_Op'] == 'Match_Size':
                        result = resize(images[j['src_im_ind']], images[j['parameters'][0]].shape)
                        result = (result*255).astype('uint8')

                elif process in morph_list:

                    # Set the kernel for morphological operation
                    

                    if j['kernel'] == 'Rectangle':
                        morph_kernel = cv.getStructuringElement(cv.MORPH_RECT,( j['parameters'][0], j['parameters'][1]))

                    elif j['kernel'] == 'Cross':
                        morph_kernel = cv.getStructuringElement(cv.MORPH_CROSS,( j['parameters'][0], j['parameters'][1]))

                    elif j['kernel'] == 'Ellipse':
                        morph_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,( j['parameters'][0], j['parameters'][1]))

                    # Apply the morphological operation

                    if process == 'Erode':
                        result = cv.erode(images[j['src_im_ind']], morph_kernel, iterations = j['parameters'][2])

                    elif process == 'Dilate':
                        result = cv.dilate(images[j['src_im_ind']], morph_kernel, iterations = j['parameters'][2])

                    elif process == 'Open':
                        result = cv.morphologyEx( images[j['src_im_ind']], cv.MORPH_OPEN, morph_kernel)

                    elif process == 'Close':
                        result = cv.morphologyEx( images[j['src_im_ind']], cv.MORPH_CLOSE, morph_kernel)

                    elif process == 'Morph_Gradient':
                        result = cv.morphologyEx( images[j['src_im_ind']], cv.MORPH_GRADIENT, morph_kernel)

                    elif process == 'Top_Hat':
                        result = cv.morphologyEx( images[j['src_im_ind']], cv.MORPH_TOPHAT, morph_kernel)

                    elif process == 'Black_Hat':
                        result = cv.morphologyEx( images[j['src_im_ind']], cv.MORPH_BLACKHAT, morph_kernel)

                images.append(result)

                if batch_stack == 'batch':
                
                    if j['im_name'] in im_save:



                        filepath = batch_folder_name + j['im_name'] + '/' + filename[0] + j['im_name'] + '.jpg'
                        cv.imwrite(filepath, result)
                print(time.time()-t3)
                
            if batch_stack == 'stack':
                try:
                    color[image_filenames.index(i),:,:,:] = base_color
                    mask[image_filenames.index(i),:,:] = result            
                except:
                    color = np.copy(base_color)
                    color = color[np.newaxis,:,:,:]
                    mask = np.copy(result)
                    mask = mask[np.newaxis,:,:]
            
            print(time.time() - t2)

        print(time.time()-t1)

        if batch_stack == 'stack':
            
            hist_max = 255
            hist_min = 10
            hist_width = hist_max - hist_min
            histograms = np.zeros((len(image_filenames), (hist_width)*3))

            for i in range(len(image_filenames)):
                for j in range(color[i].shape[3]):
                    hist, bin_edges = np.histogram(color[i,:,:,j], bins=np.arange(257))
                    histograms[i,(j*hist_width):(j*hist_width+hist_width)] = hist[hist_min:hist_max]
                print(i, '  ', time.time()-t1)

            return color, mask, histograms