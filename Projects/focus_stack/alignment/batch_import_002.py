import cv2 as cv
import os

def get_images(directory):
    a = os.walk(directory)
    for root, dirs, files in a:
        b = files
    
    images = []
    
    for i in range(len(b)):
        images.append(cv.imread(directory+b[i], cv.IMREAD_COLOR))
        
    
    return images


directory = '/Users/anthonyesposito/Pictures/macroni/Rosasite_w_Conacalcite/1/JPG/'

images = get_images(directory)

for i in range(len(images)):
    cv.imshow('window', images[i])
    cv.waitKey(100)