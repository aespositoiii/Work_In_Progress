import cv2 as cv
import numpy as np

src = cv.imread('/Users/anthonyesposito/Pictures/macroni/Rosasite_w_Conacalcite/1/JPG/DSCF6941.jpg')
src2 = cv.imread('/Users/anthonyesposito/Pictures/macroni/Rosasite_w_Conacalcite/1/JPG/DSCF6942.jpg')
srcgray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
src2gray = cv.cvtColor(src2, cv.COLOR_BGR2GRAY)
#src = src.astype('float64')
#src2 = src2.astype('float64')
#srcgray = srcgray.astype('float64')
#src2gray = src2gray.astype('float64')

print(src.dtype)
print(srcgray.dtype)
cv.imshow('a window', src)
cv.waitKey(500)
cv.imshow('another window', srcgray)
cv.waitKey(500)
cv.imshow('an other window', src2)
cv.waitKey(500)
im_reg_mapper = cv.reg_MapperGradEuclid()
c = src2
for i in range(100):
    
    a = im_reg_mapper.calculate(src, c)
    c = a.inverseWarp(c,c)

    if i%1 == 0:
        print(i)
        ind = 0
        print('original')
        for j in range(10):
            ind = (ind+1) % 2

            if ind == 0:
                cv.imshow("Image", src)
                cv.waitKey(1)
            else:
                cv.imshow("Image", src2 )
                cv.waitKey(1)
        ind = 0
        
        print('warped')
        for j in range(10):
            ind = (ind+1) % 2

            if ind == 0:
                cv.imshow("Image", src)
                cv.waitKey(1)
            else:
                cv.imshow("Image", c )
                cv.waitKey(1)

print(a, '\n\n', type(a))
print(c.shape, '\n\n', type(c), '\n\n', src.shape)
cv.imshow('another other window', (src+c) // 2 )



