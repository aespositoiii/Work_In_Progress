import numpy as np
import cv2 as cv
from scipy import stats

def log_gabor(img, fo, sigmaOnf, angle, thetaSigma):
    
    if len(img.shape) == 3:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    #fo = 1.0/wavelength
    
    rows, cols = img.shape

    cols_opt = cv.getOptimalDFTSize(cols)
    rows_opt = cv.getOptimalDFTSize(rows)

    img_zeros_opt = np.zeros((rows_opt, cols_opt), dtype='float32')
    img_zeros_opt[:rows, :cols] = img
    image = np.zeros(img_zeros_opt.shape)
    image = img_zeros_opt
    #image = img

    xv = np.linspace(-cols_opt//2+1, cols_opt//2, cols_opt)
    yv = np.linspace(-rows_opt//2+1, rows_opt//2, rows_opt)

    #xv = np.linspace(-cols//2+1, cols//2, cols)
    #yv = np.linspace(-rows//2+1, rows//2, rows)

    x, y = np.meshgrid(xv, yv)

    double_zero = np.where((x==0) & (y==0))

    x[double_zero[0][0], double_zero[1][0]] = 1
    y[double_zero[0][0], double_zero[1][0]] = 1

    radius = np.sqrt(np.copy(x)**2 + np.copy(y)**2)

    logGabor = np.exp( - ( np.log( radius / fo ) ) ** 2 ) / (2 * np.log( sigmaOnf ) ** 2 )
    logGabor[double_zero[0][0], double_zero[1][0]] = 0

    angle = angle * np.pi

    theta = np.arctan2(np.copy(-y),np.copy(x))
    sintheta = np.sin(theta)
    costheta = np.cos(theta)

    ds = sintheta * np.cos(angle) - costheta * np.sin(angle)
    dc = costheta * np.cos(angle) + sintheta * np.sin(angle)

    dtheta = np.abs(np.arctan2(ds,dc))
    spread = np.exp( ( -dtheta ** 2 ) / ( 2 * thetaSigma ** 2 ))

    log_Gabor_filter = spread * logGabor
    #log_Gabor_filter = log_Gabor_filter/log_Gabor_filter.max()

    LG_shift = np.fft.ifftshift(log_Gabor_filter)

    img_F = cv.dft(np.float32(image))
    img_F_LG = np.zeros(img_F.shape, np.dtype('float32'))
    img_F_LG = cv.multiply(img_F, LG_shift, img_F_LG, dtype=5)
    img_back = cv.idft(img_F_LG)
    img_back = img_back[:rows, :cols]



    img_back = img_back / np.max([np.abs(img_back.max()), np.abs(img_back.min())])
    
    img_back = img_back - img_back.min()
    
    img_back = img_back / img_back.max()
    
    img_back = (255 * img_back).astype('uint8')

    
    mode, count = stats.mode(img_back, axis=None)

    mode = mode.astype('uint8')

     

    img_back[:18, :] = mode
    img_back[:, :18] = mode
    img_back[-17: , :] = mode
    img_back[: , -17:] = mode
    
    return img_back, log_Gabor_filter