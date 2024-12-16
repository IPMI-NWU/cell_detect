#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2020/6/15 19:35
# @Author  : Can Cui
# @File    : compare_image.py
# @Software: PyCharm
# @Comment:
from . import  imagehash
from PIL import Image
import numpy as np

def compute_hash(image, num_crop=2):
    '''
    :param image:
    :param num_crop:  crop to num_crop * num_crop patches
    :return: list of hash code of patches
    '''
    hash_list = []
    h,w = image.shape[0:2]
    crop_h, crop_w = h//num_crop, w//num_crop
    for i in range(num_crop):
        for j in range(num_crop):
            crop_patch = image[i*crop_h:(i+1)*crop_h, j*crop_w:(j+1)*crop_w, :]
            crop_patch_image = Image.fromarray(crop_patch)
            hash_list.append(imagehash.average_hash(crop_patch_image).hash)
    return  hash_list


def compute_similarity(ori_hash, comp_hash, num_crop=2):
    if len(ori_hash) == len(comp_hash):
        score_list = []
        num_hash = num_crop * num_crop
        for i in range(num_hash):
            o_hash_code = ori_hash[i]
            c_hash_code = comp_hash[i]
            oc_score = len(np.where((o_hash_code == c_hash_code) == True)[0]) / o_hash_code.size
            score_list.append(oc_score)
        scores_np = np.array(score_list)
        score = np.min(scores_np)
    else:
        score = 0
    return score


# def compute_similarity_with_pre(pre_hash, target_image, num_crop=2):
#     c_hash_code = compute_hash(target_image, num_crop=num_crop)
#     compute_similarity(pre_hash, c_hash_code, num_crop=num_crop)



