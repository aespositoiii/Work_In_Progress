from stacking_funcs_015 import registration, mask_blur, img_warp
import cv2 as cv
import numpy as np
import time

t0 = time.time()

im1 =  cv.imread("/Users/anthonyesposito/Pictures/macroni/Rosasite_w_Conacalcite/1/JPG/DSCF6941.jpg")
im2 =  cv.imread("/Users/anthonyesposito/Pictures/macroni/Rosasite_w_Conacalcite/1/JPG/DSCF6942.jpg")

im1_grad = mask_blur(img=im1, thresh=15, n_iter=50) + 1
im2_grad = mask_blur(img=im2, thresh=15, n_iter=50)

warp_matrix = registration(im1, im2, warp_mode=cv.MOTION_EUCLIDEAN, number_of_iterations=1000, termination_eps=1e-3)

im2 = img_warp(im2, warp_matrix, warp_mode=cv.MOTION_EUCLIDEAN)
im2_grad_warp = img_warp(im2_grad, warp_matrix, warp_mode=cv.MOTION_EUCLIDEAN)+1

exp = 2

im1_grad_float = im1_grad.astype('float32') ** exp
im2_grad_warp_float = im2_grad_warp.astype('float32') ** exp

im1_mask = (im1_grad_float) / (im2_grad_warp_float + im1_grad_float)
im2_mask_warp = (im2_grad_warp_float) / (im2_grad_warp_float + im1_grad_float)

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
cv.waitKey(1000)