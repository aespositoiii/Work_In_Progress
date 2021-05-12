import numpy as np
import cv2 as cv

def log_gabor(img, wavelength, sigmaOnf, angle, thetaSigma):

    if len(img.shape) == 3:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)


    fo = 1.0/wavelength
    
    cols = img.shape[1]
    rows = img.shape[0]
    print(cols, rows)

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

    img_F = np.fft.fft2(img)
    img_F_LG = img_F * LG_norm_shift
    img_back = np.fft.ifft2(img_F_LG)
    img_back = np.real(img_back)

    return img_back, LG_norm