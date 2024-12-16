#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2020/9/2 15:41
# @Author  : Can Cui
# @File    : semi_manual_thread.py
# @Software: PyCharm
# @Comment:

import cv2
import numpy as np
from functools import partial
from threading import Thread
from sklearn.cluster import KMeans
from scipy import ndimage


'''
Functions bellow are multiprocessing distinguish
'''
# '''

def processed_shade(R, mask_label, average_variance_list, average_shade_list, center_coords, coord_shade_dict):

    # print('--------------------------start--------------------------------')
    std_var = 200
    for i in range(len(center_coords)):
        coord = center_coords[i]

        index = mask_label[coord[1], coord[0]]
        var = average_variance_list[index]
        if var > std_var:
            X = R[np.where(mask_label == index)]
            k_means = KMeans(n_clusters=2, random_state=0).fit(X.reshape(-1, 1))
            X_labels = k_means.labels_
            shade = min(np.sum(X_labels * X)/np.sum(X_labels),
                        np.sum((1-X_labels) * X)/np.sum((1-X_labels)))
            coord_shade_dict[coord] = shade
        else:
            shade = average_shade_list[index]
            coord_shade_dict[coord] = shade

def filter(center_label_dict):
    return {k: v for k, v in center_label_dict.items() if (v == 2)or(v == 5)}

def split_list(listTemp, n):
    for i in range(0, len(listTemp), n):
        yield listTemp[i:i + n]

def semi_auto_distinguish(seed, center_label_dict, standard_shade_value, R):

    standard_shade = standard_shade_value
    tumor_seed = seed[:, :, 2] + seed[:, :, 5]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    seed_dilated = cv2.dilate(tumor_seed, kernel, iterations=1)
    mask_label, num_regions = ndimage.label(seed_dilated)

    index = np.unique(mask_label)
    average_shade_list = ndimage.mean(R, labels=mask_label, index=index)
    average_variance_list = ndimage.variance(R, labels=mask_label, index=index)

    tumor_center_dict = filter(center_label_dict)
    center_coord_list = list(tumor_center_dict.keys())

    coord_shade_dict = dict()
    split_center_coord = split_list(center_coord_list, max(1, len(center_coord_list)// 10))
    process_list = []

    for center_coords in split_center_coord:
        # p = multiprocessing.Process(target=partial(processed_shade, R, mask_label, average_variance_list, average_shade_list),
        #             args=(center_coords, coord_shade_dict))

        p = Thread(target=partial(processed_shade, R, mask_label, average_variance_list, average_shade_list),
                    args=(center_coords, coord_shade_dict))

        process_list.append(p)

    for p in process_list:
        p.start()

    for res in process_list:
        res.join()

    for key in coord_shade_dict.keys():
        center_coord = key
        shade = coord_shade_dict[center_coord]
        if shade <= standard_shade:
            center_label_dict[center_coord] = 5
        else:
            center_label_dict[center_coord] = 2

    center_coords = np.array(list(center_label_dict)).astype(np.int)
    labels = list(center_label_dict.values())
    labels = np.array(labels).astype(np.int)

    return center_coords, labels
# '''