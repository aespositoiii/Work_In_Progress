from __future__ import print_function
import cv2 as cv
import numpy as np



def alignImages(im1, im2):

    MAX_FEATURES = 5000
    
    GOOD_MATCH_PERCENT = 0.2
    
    # Convert images to grayscale
    im1Gray = cv.cvtColor(im1, cv.COLOR_BGR2GRAY)
    im2Gray = cv.cvtColor(im2, cv.COLOR_BGR2GRAY)

    #im1Gray = cv.GaussianBlur(im1Gray, (3, 3), 0)
    #im2Gray = cv.GaussianBlur(im2Gray, (3, 3), 0)

    # Detect ORB features and compute descriptors.
    orb = cv.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

    # Match features.
    matcher = cv.DescriptorMatcher_create(cv.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    # Draw top matches
    imMatches = cv.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
    cv.imwrite("matches.jpg", imMatches)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv.findHomography(points1, points2, cv.RANSAC)

    # Use homography
    height, width, channels = im2.shape
    im1Reg = cv.warpPerspective(im1, h, (width, height))

    return im1Reg, h
'''
if __name__ == '__main__':

    # Read reference image
    refFilename = "form.jpg"
    print("Reading reference image : ", refFilename)
    imReference = cv.imread(refFilename, cv.IMREAD_COLOR)

    # Read image to be aligned
    imFilename = "scanned-form.jpg"
    print("Reading image to align : ", imFilename);
    im = cv.imread(imFilename, cv.IMREAD_COLOR)

    print("Aligning images ...")
    # Registered image will be resotred in imReg.
    # The estimated homography will be stored in h.
    imReg, h = alignImages(im, imReference)

    # Write aligned image to disk.
    outFilename = "aligned.jpg"
    print("Saving aligned image : ", outFilename);
    cv.imwrite(outFilename, imReg)

    # Print estimated homography
    print("Estimated homography : \n",  h)
'''

#src = cv.imread('/Users/anthonyesposito/Pictures/macroni/Rosasite_w_Conacalcite/1/JPG/DSCF6941.jpg')
#src2 = cv.imread('/Users/anthonyesposito/Pictures/macroni/Rosasite_w_Conacalcite/1/JPG/DSCF6942.jpg')
src = cv.imread('/Users/anthonyesposito/Pictures/macroni/5_4_2021_test/JPG/DSCF8758.jpg')
src2 = cv.imread('/Users/anthonyesposito/Pictures/macroni/5_4_2021_test/JPG/DSCF8759.jpg')
print('a')
srcreg, h = alignImages(src, src2)

ind = 0
print('original')
for j in range(10):
    print(j)
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
    print(j)
    ind = (ind+1) % 2

    b=500

    if ind == 0:
        cv.imshow("Image", srcreg)
        cv.waitKey(1)
    else:
        cv.imshow("Image", src2 )
        cv.waitKey(1)