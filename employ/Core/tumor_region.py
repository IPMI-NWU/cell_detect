#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2020/9/2 18:25
# @Author  : Can Cui
# @File    : tumor_region.py
# @Software: PyCharm
# @Comment:

import numpy as np
import scipy.ndimage as ndimage
from scipy.misc import imsave
from skimage.draw import polygon
import time
import cv2

def cnts2mask(cnts, contour_type, h, w, downscale=4):
    '''
    :param cnts: [
                    [[x1,y1], [x2,y2], ...], #contour1
                    [[x1,y1], [x2,y2], ...], #contour2
                               ...
                 ]

    :return:
    '''
    contour_type = int(contour_type)

    i = 0
    mask = np.zeros((h//downscale, w//downscale))
    for cnt in cnts:
        x_coords, y_coords = zip(*cnt)

        x_coords = np.array(x_coords)//4
        y_coords = np.array(y_coords)//4
        i += 1
        rr, cc = polygon(y_coords, x_coords, mask.shape)
        mask[rr,cc] = 1

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = ndimage.binary_fill_holes(mask).astype(np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.resize(mask, (w, h))
    if contour_type ==0: # reverse region
        mask = mask.astype(np.bool)
        mask = ~mask
    else:
        pass

    return mask.astype(np.uint8)





