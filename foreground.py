from __future__ import print_function
import cv2
import numpy as np
from openslide import OpenSlide

def read_region(svs, x, y, level, size, flip_channels=False, verbose=False):
    # Utility function because openslide loads as RGBA
    if verbose:
        print('Reading SVS: ({},{}), LEVEL {}, SIZE={}'.format(
            x,y,level,size))
    #/end if

    ## TODO Check if region is out of range for the requested level
    # level_dims = svs.level_dimensions[level]
    # assert x > 0 and y > 0, print 'data_utils.read_region: X and Y must be positive'
    # ## Not sure of the order
    # assert x + size[0] < level_dims[1]
    # assert y + size[1] < level_dims[0]

    img = svs.read_region((x,y), level, size)
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    if flip_channels:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) ## This actually needs to be here

    return img


def read_low_level(svs, low_level_index=None, verbose=False):
    if low_level_index is None:
        low_index = svs.level_count - 1
    else:
        low_index = low_level_index

    img = read_region(svs, 0, 0, low_index, svs.level_dimensions[low_index],
        flip_channels=True, verbose=verbose)
    return img


def whitespace(img, mode='Otsu', white_pt=225):
    if len(img.shape)==3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    if mode=='Otsu':
        bcg_level, img = cv2.threshold(img, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    elif mode=='thresh':
        img = (img < white_pt).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    else:
        raise ValueError('foreground:whitespace mode must be "Otsu" or "thresh"')

    return img > 0


def get_process_map(masks):
    # Inverse the masks and take the union, by adding them together.
    inv = lambda x: 1 - x
    masks = [inv(mask) for mask in masks]
    n_masks = len(masks)
    if n_masks == 1:
        mask = masks[0]
    elif n_masks > 1:
        mask = np.add(masks)

    mask = mask == n_masks
    return mask


"""
https://www.learnopencv.com/filling-holes-in-an-image-using-opencv-python-c/
"""
def imfill(img):
    if img.dtype == 'bool':
        img = img.astype(np.uint8)

    ## Old way:
    # https://www.learnopencv.com/filling-holes-in-an-image-using-opencv-python-c/
    # open cv contours
    if cv2.__version__[0]=='3':
        _, cnts, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    else:
        cnts, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    hulls = [cv2.convexHull(cnt) for cnt in cnts if cv2.contourArea(cnt) > 2000]
    img2 = np.zeros_like(img)

    cv2.drawContours(img2, hulls, -1, (1), -1)

    return img2 > 0


def get_foreground(svs, low_level_index=None):
    img = read_low_level(svs, low_level_index=low_level_index)

    # Boolean image of white areas
    whitemap = whitespace(img, mode='thresh')

    whitemap_filled = imfill(whitemap)

    ## Really shouldn't need this
    if whitemap_filled.dtype == 'bool':
        process_map = whitemap_filled.astype(np.uint8)
    elif whitemap_filled.dtype == 'uint8':
        process_map = whitemap_filled

    return process_map
