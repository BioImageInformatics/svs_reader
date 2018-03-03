"""
Color transfer of a reference LAB vector mean & standard deviation.

E. Reinhard "Color transfer between images" <em>IEEE Comput. Graph. Appl.</em> vol. 21 no. 5 pp. 34-41 Sep./Oct. 2001.

image
    3-channel RGB space image
target
    3x2 ndarray float64
"""

import numpy as np
import cv2

def reinhard(image, target=None):
    ## Default target, as in Gertych, et al 2015
    if target is None:
        target = np.array([[148.60, 41.56], [169.30, 9.01], [105.97, 6.67]])

    M, N = image.shape[:2]

    whitemask = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    whitemask = whitemask > 215 ## TODO: Hard code threshold; replace with Otsu

    imagelab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    imageL, imageA, imageB = cv2.split(imagelab)

    # mask is valid when true
    imageLM = np.ma.MaskedArray(imageL, whitemask)
    imageAM = np.ma.MaskedArray(imageA, whitemask)
    imageBM = np.ma.MaskedArray(imageB, whitemask)

    ## Sometimes STD is near 0, or 0; add epsilon to avoid div by 0 -NI
    epsilon = 1e-8

    imageLMean = imageLM.mean()
    imageLSTD = imageLM.std() + epsilon

    imageAMean = imageAM.mean()
    imageASTD = imageAM.std() + epsilon

    imageBMean = imageBM.mean()
    imageBSTD = imageBM.std() + epsilon

    # normalization in lab
    imageL = (imageL - imageLMean) / imageLSTD * target[0][1] + target[0][0]
    imageA = (imageA - imageAMean) / imageASTD * target[1][1] + target[1][0]
    imageB = (imageB - imageBMean) / imageBSTD * target[2][1] + target[2][0]

    imagelab = cv2.merge((imageL, imageA, imageB))
    imagelab = np.clip(imagelab, 0, 255)
    imagelab = imagelab.astype(np.uint8)

    # Back to RGB space
    returnimage = cv2.cvtColor(imagelab, cv2.COLOR_LAB2RGB)
    # Replace white pixels
    returnimage[whitemask] = image[whitemask]

    return returnimage
