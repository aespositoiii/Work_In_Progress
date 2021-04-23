from stacking_funcs_014 import registration, mask_blur
import cv2 as cv
import numpy as np
import time

im1 =  cv.imread("/Users/anthonyesposito/Pictures/macroni/Rosasite_w_Conacalcite/1/JPG/DSCF6941.jpg");
im2 =  cv.imread("/Users/anthonyesposito/Pictures/macroni/Rosasite_w_Conacalcite/1/JPG/DSCF6942.jpg");

registration(im1,im2, warp_mode=cv.MOTION_EUCLIDEAN, number_of_iterations=1000, termination_eps=1e-3)

mask = mask_blur(img=im1, thresh=15, n_iter=50)

cv.imshow('window', mask)
cv.waitKey(5000)

mask = mask_blur(img=im2, thresh=15, n_iter=50)

cv.imshow('window', mask)
cv.waitKey(5000)