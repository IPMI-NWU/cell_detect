#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2020/6/15 16:57
# @Author  : Can Cui
# @File    : semi_cell_seg.py
# @Software: PyCharm
# @Comment:


import numpy as np
import cv2
import time
from scipy import ndimage
import torch
from  skimage.feature import peak_local_max
from .voting import voting
from .preprocess import adjust_contrast
from .semi_manual_thread import semi_auto_distinguish

import torch.nn.functional as F
from torchvision import transforms


def split_image_cuda(image, patch_size, overlap):

    h,w = image.size()[0:2]
    stride = patch_size - overlap
    patch_list = []
    num_y, num_x = 0,0

    for y in range(0, h, stride):
        num_x = 0
        for x in range(0, w, stride):
            crop_img = image[y:y+patch_size, x:x+patch_size, :]
            crop_h, crop_w = crop_img.size()[0:2]
            pad_h, pad_w = patch_size-crop_h, patch_size-crop_w
            crop_img = crop_img.permute(2,0,1).unsqueeze(0)
            if pad_h>0 or pad_w>0:
                crop_img = torch.nn.functional.pad(crop_img, (0, pad_w ,0, pad_h), 'constant', 255)
            patch_list.append(crop_img)
            num_x+=1
        num_y+=1
    patch_image = torch.cat(patch_list)
    return patch_image, num_y, num_x

def reconstruct_mask_cuda(masks,  patch_size, overlap, num_y, num_x):
    num_channel = masks.shape[1]
    stride = patch_size - overlap
    mask_h, mask_w = patch_size+(num_y-1)*stride, patch_size+(num_x-1)*stride
    result_mask = torch.zeros((num_channel, mask_h, mask_w)).cuda()
    mask_count = torch.zeros((mask_h, mask_w, 1)).cuda()

    for y in range(num_y):
        for x in range(num_x):
            i = y*num_x + x
            ys, ye = y*stride, y*stride+patch_size
            xs, xe = x*stride, x*stride+patch_size
            result_mask[:, ys:ye, xs:xe] += masks[i]
            mask_count[ys:ye, xs:xe, :] += 1
    result_mask = result_mask.permute(1,2,0)
    result_mask /= mask_count
    return result_mask


def generate_result_mask_cuda(image, net, patch_size=512, overlap=128, batch_size=16):
    with torch.no_grad():
        img_tensor = torch.from_numpy(image).float().cuda()
        img_h, img_w = img_tensor.size()[0:2]
        patch_tensor, num_y, num_x = split_image_cuda(img_tensor, patch_size,  overlap)
        num_patches = patch_tensor.size()[0]
        patch_tensor = patch_tensor * (2. / 255) - 1.
        results = []

        for i in range(0, num_patches, batch_size):
            this_batch = patch_tensor[i:i + batch_size]
            result = net(this_batch)
            sigmoid_result = torch.sigmoid(result[:, :7, :, :])
            alpha = 0.25
            sigmoid_result[:, 6, :, :] *= alpha
            results.append(sigmoid_result)

        results = torch.cat(results)
        result_masks = reconstruct_mask_cuda(results, patch_size, overlap, num_y, num_x)
        result_masks = result_masks[0:img_h, 0:img_w, :]
        return result_masks

def get_coordinate(voting_map, min_len=6):
    # voting_map = cv2.getGaussianKernel()
    voting_map  = cv2.GaussianBlur(voting_map,(49,49),cv2.BORDER_DEFAULT)

    coordinates = peak_local_max(voting_map, min_distance=min_len, indices=True, exclude_border=min_len // 2)  # N by 2
    if coordinates.size == 0:
        coordinates = None
        return coordinates

    boxes_list = [coordinates[:, 1:2], coordinates[:, 0:1], coordinates[:, 1:2], coordinates[:, 0:1]]
    coordinates = np.concatenate(boxes_list, axis=1)
    return coordinates


def post_process_mask(masks, tumor_mask, threshold, resize_ratio):

    voting_map = torch.sum(masks, dim=2)
    voting_map[voting_map<threshold*torch.max(voting_map)] = 0

    if tumor_mask is not None:
        tumor_mask = cv2.resize(tumor_mask, (voting_map.shape[1], voting_map.shape[0]))
        tumor_mask = torch.from_numpy(tumor_mask).cuda()
        voting_map[tumor_mask==0]=0

    bboxes = get_coordinate(voting_map.cpu().numpy(), min_len = int(10*resize_ratio))

    if bboxes is None:
        return None, None

    x_coords = bboxes[:,0]
    y_coords = bboxes[:,1]
    pred_center_coords = bboxes[:, 0:2]
    label_map = torch.argmax(masks, dim=2).cpu().numpy()
    pred_label = label_map[y_coords, x_coords]

    return  pred_center_coords, pred_label


def get_standard_shade(image,  x, y):

    if x>0 and y >0:
        B, G, R = cv2.split(image)
        std_seed = np.zeros_like(R)
        std_seed[int(y), int(x)] = 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        std_seed = cv2.dilate(std_seed, kernel, iterations=1)

        mask_label, _ = ndimage.label(std_seed)
        index = np.unique(mask_label)
        standard_shade = ndimage.mean(R, labels=mask_label, index=index)[1]
    else:
        standard_shade = -1
    return  standard_shade


def cal_ki67_np(ori_img,net, standard_shade_value, tumor_mask=None, contrast_value=50, resize_ratio=1):
    img = ori_img.copy()
    h, w = img.shape[0:2]

    if resize_ratio != 1:
        resized_image = cv2.resize(img, (int(w*resize_ratio), int(h*resize_ratio)))
    else:
        resized_image = ori_img

    enhanced_img = adjust_contrast(resized_image, contrast_val=contrast_value)

    _, _, R = cv2.split(enhanced_img)

    result_masks = generate_result_mask_cuda(enhanced_img, net, patch_size=512, overlap=64, batch_size= 4)
    coords, labels = post_process_mask(masks=result_masks, tumor_mask=tumor_mask, threshold=0.1, resize_ratio=resize_ratio)

    if coords is None:
        return None, None

    result_masks = result_masks.cpu().numpy()


    # if tumor_mask is None:
    #     vote lymph cells to tumor cells
        # coords_label_dict = dict(zip(tuple(map(tuple, coords)), labels))
        # coords_label_dict, coords, labels = \
        #     voting(resized_image, R, result_masks, coords_label_dict, coords, labels)
    # else:
    labels[labels == 0] = 2  # negative fibre to negative tumor
    labels[labels == 1] = 2  # negative lymph to negative tumor
    labels[labels == 6] = 2  # other cells to negative tumor
    labels[labels == 4] = 5  # positive lymph to postive tumor

    coords_label_dict = dict(zip(tuple(map(tuple, coords)), labels))

    seed = np.zeros_like(result_masks)
    x_coords = coords[:, 0]
    y_coords = coords[:, 1]
    seed[y_coords, x_coords, labels] = 1

    if standard_shade_value > 0:
        # adjust positive and negative tumor cells based on manual shade guidance
        coords, labels = \
            semi_auto_distinguish(seed, coords_label_dict, standard_shade_value, R)
    else:
        pass

    #remove positive fibre cells
    coords = coords[labels!=3]
    labels = labels[labels!=3]

    coords = (coords/resize_ratio).astype(int)
    return coords, labels


def max_confidence(points, scores, distance_threshold):
    n = len(points)
    fused = np.full(n, False)
    result = np.zeros((0, 2))
    classes = np.array([], dtype=int)
    for i in range(n):
        if not fused[i]:
            fused_index = np.where(np.linalg.norm(points[[i]] - points[i:], 2, axis=1) < distance_threshold)[0] + i
            fused[fused_index] = True

            r_, c_ = np.where(scores[fused_index] == np.max(scores[fused_index]))
            r_, c_ = [r_[0]], [c_[0]]
            result = np.append(result, points[fused_index[r_]], axis=0)
            classes = np.append(classes, c_)
    return result.astype(int), classes.astype(int)


@torch.no_grad()
def cal_pdl1_np(ori_img, net, thresh=0.5):
    trans = transforms.Compose([transforms.ToTensor()])
    pic = trans(ori_img)

    mean, std = np.load('mean_std.npy')
    for t, m, s in zip(pic, mean, std):
        t.sub_(m).div_(s)

    pic = pic.unsqueeze(0)
    pic = pic.cuda()
    output = net(pic)

    fore_scores = F.softmax(output['fore_logits'][0], dim=-1)
    fore_points = fore_scores[:, 1] > thresh
    outputs_scores = F.softmax(output['pred_logits'][0][fore_points], dim=-1).cpu().numpy()
    outputs_points = output['pred_points'][0][fore_points].cpu().numpy()

    coords, labels = max_confidence(outputs_points, outputs_scores, 12)

    return coords, labels
