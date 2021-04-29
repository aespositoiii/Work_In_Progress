from stacking_funcs_019 import registration, mask_blur, img_warp, laplace_threshold
import cv2 as cv
import numpy as np
import time

t0 = time.time()

im1 =  cv.imread("/Users/anthonyesposito/Pictures/macroni/Rosasite_w_Conacalcite/1/JPG/DSCF6941.jpg")
im2 =  cv.imread("/Users/anthonyesposito/Pictures/macroni/Rosasite_w_Conacalcite/1/JPG/DSCF6942.jpg")

im1_lap, norm1 = laplace_threshold(im1, 20, 11)
im1_lap = cv.normalize(im1_lap, im1_lap, 0, 255, cv.NORM_MINMAX)
im2_lap, norm2 = laplace_threshold(im2, 20, 11)
im2_lap= cv.normalize(im2_lap, im2_lap, 0, 255, cv.NORM_MINMAX)

'''cv.imshow('im1_lap', im1_lap)
cv.waitKey(1)
cv.imshow('im2_lap', im2_lap)
cv.waitKey(0)'''


kernel = np.ones((5,5), np.uint8)
  
# The first parameter is the original image,
# kernel is the matrix with which image is 
# convolved and third parameter is the number 
# of iterations, which will determine how much 
# you want to erode/dilate a given image. 
n_iter = 5
n_cycles = 100

for i in range(n_cycles):
    im1_dilation = cv.dilate(im1_lap, kernel, iterations=n_iter)
    im1_erosion = cv.erode(im1_dilation, kernel, iterations=n_iter)
for i in range(n_cycles):
    im2_dilation = cv.dilate(im2_lap, kernel, iterations=n_iter)
    im2_erosion = cv.erode(im2_dilation, kernel, iterations=n_iter)


cv.imshow('im1_erosion', im1_erosion)
cv.waitKey(1)
cv.imshow('im2_erosion', im2_erosion)
cv.waitKey(0)


'''im1_grad = mask_blur(img=im1, thresh=15, n_iter=50) + 1
im2_grad = mask_blur(img=im2, thresh=15, n_iter=50)

warp_matrix = registration(im1, im2, warp_mode=cv.MOTION_EUCLIDEAN, number_of_iterations=1000, termination_eps=1e-3)

im2 = img_warp(im2, warp_matrix, warp_mode=cv.MOTION_EUCLIDEAN)
im2_grad_warp = img_warp(im2_grad, warp_matrix, warp_mode=cv.MOTION_EUCLIDEAN)+1

im1_mask = (im1_grad.astype('float32')**3) / (im2_grad_warp.astype('float32')**3 + im1_grad.astype('float32')**3)
im2_mask_warp = (im2_grad_warp.astype('float32')**3) / (im2_grad_warp.astype('float32')**3 + im1_grad.astype('float32')**3)

ind=0

im1_masked = np.copy(im1)
im2_masked = np.copy(im2)

for i in range(im1.shape[2]):
    im1_masked[:,:,i] = im1[:,:,i] * im1_mask
    im2_masked[:,:,i] = im2[:,:,i] * im2_mask_warp

comb = im1_masked+im2_masked

print(time.time()-t0)

for i in range(30):
    if ind==0:
        cv.imshow('window', im1)
        cv.waitKey(1)
        ind=1

    elif ind==1:
        cv.imshow('window', im2)
        cv.waitKey(1)
        ind=0

cv.imshow('window', comb)
cv.waitKey(1000)'''