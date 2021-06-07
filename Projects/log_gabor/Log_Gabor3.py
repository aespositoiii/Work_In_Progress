import numpy as np
import cv2 as cv

def log_gabor3(img, wavelength, sigmaOnf, angle, thetaSigma):

    if img.shape[2] == 3:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    fo = 1.0/wavelength
    
    rows, cols = img.shape

    cols_opt = cv.getOptimalDFTSize(cols)
    rows_opt = cv.getOptimalDFTSize(rows)

    img_zeros_opt = np.zeros(rows_opt, cols_opt)
    img_zeros_opt[:rows, :cols] = img

    xv = np.linspace(-cols//2+1, cols//2, cols)
    yv = np.linspace(-rows//2+1, rows//2, rows)

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
    LG_norm = log_Gabor_filter / log_Gabor_filter.max()

    LG_norm_shift = np.fft.ifftshift(LG_norm)

    img_F = cv.dft(np.float32(imgG))
    img_F_LG = np.zeros(img_F.shape, np.dtype('float32'))
    img_F_LG = cv.multiply(img_F, LG_norm_shift, img_F_LG, dtype=5)
    img_back = cv.idft(img_F_LG)

    return img_back