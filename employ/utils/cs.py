#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2021/12/19 19:56
# @Author  : Can Cui
# @File    : cs.py
# @Software: PyCharm
# @Comment:

from Slide import openSlide
import numpy as np
import random
import imageio
from PIL import Image
import os, json


multi_cell_cls_dict = {
    'neg':0,
    'ASCUS': 1,
    'LSIL': 2,
    'HSIL': 3,
    'ASC-H': 4,
    'AGC': 5
}
multi_microorganism_cls_dict = {
    'neg':0,
    '放线菌': 1,
    '滴虫': 2,
    '霉菌': 3,
    '疱疹': 4,
    '线索': 5
}
multi_cell_cls_dict_reverse ={v:k for k,v in multi_cell_cls_dict.items()}
multi_microorganism_cls_dict_reverse = {v:k for k,v in multi_microorganism_cls_dict.items()}


def process_cell_result(result):
    '''
    :param result:
    :return:
    '''
    cell_list = []

    microbe_bboxes = result['microbe_bboxes1']
    microbe_pred = result['microbe_pred1']
    cell_bboxes = result['cell_bboxes1']
    cell_prob = result['cell_prob1']

    pick_idx_M = np.where(microbe_pred==multi_microorganism_cls_dict['霉菌'])[0][:5]
    pick_idx_XS = np.where(microbe_pred==multi_microorganism_cls_dict['线索'])[0][:3]
    pick_idx_T = np.where(microbe_pred==multi_microorganism_cls_dict['滴虫'])[0][:4]
    picked_microbe_idx = np.hstack([pick_idx_M, pick_idx_XS, pick_idx_T])


    microbe_idx_list = list(np.hstack([picked_microbe_idx, np.setdiff1d(np.arange(microbe_bboxes.shape[0]), picked_microbe_idx)]).astype(np.int))
    cell_idx_list = list(np.arange(cell_bboxes.shape[0]).astype(np.int))

    for i in range(min(100, len(cell_idx_list)+len(microbe_idx_list))):
        if len(microbe_idx_list) > 0 and (i%7>4 or len(cell_idx_list)==0):
            cur_idx = microbe_idx_list.pop(0)
            cur_box = microbe_bboxes[cur_idx]
            # import pdb; pdb.set_trace()
            cur_label = multi_microorganism_cls_dict_reverse[int(microbe_pred[cur_idx])]
            cur_type = int(microbe_pred[cur_idx])
        else:
            cur_idx = cell_idx_list.pop(0)
            cur_box = cell_bboxes[cur_idx]
            cur_label = str(round(cell_prob[cur_idx] * 100, 2)) + "%"
            cur_type=7

        xmin, ymin, xmax, ymax = map(int, cur_box.tolist())
        roi_center  = [int((xmin+xmax)/2), int((ymin+ymax)/2)]
        contourPoint = [[xmin, ymin],[xmax, ymin], [xmax, ymax], [xmin, ymax], [xmin, ymin]]
        cell_list.append(
            {'contourPoints':contourPoint, 'type':cur_type, 'center':roi_center, 'label':cur_label}
        )
    return cell_list

def save_roi(slide_path, result, res_path, num_row=6, num_col=7, patch_size=2048, scale=1, border_thickness=15,
             is_save_label=False, label_size=500, alg_type='lct'):
    patch_size *= scale
    result_dict = {}

    num_patch = num_row * num_col
    slide = openSlide(slide_path)

    slide_mpp = slide.mpp
    cells = result['cells']
    num_cells = len(cells)

    if num_cells == 0:
        return None

    H, W = slide.height, slide.width
    h, w = (num_row + 1) * border_thickness + num_row * patch_size, \
           (num_col + 1) * border_thickness + num_col * patch_size
    combined_image = np.ones((h, w, 3), dtype=np.uint8) * 255

    patch_cells = []

    cur_roi_indx = 0  # current index in N rois
    cur_cell_idx = 0  # current index in total num of cells

    while cur_roi_indx < num_patch and cur_cell_idx < num_cells:
        try:
            this_cell = cells[cur_cell_idx]
            center_coords = this_cell['center']
            contour_coords = this_cell['contourPoints']
            if 'label' in this_cell:
                label = this_cell['label']
            else:
                label = ''
            x_center = int(center_coords[0])
            y_center = int(center_coords[1])
            crop_x, crop_y = x_center - patch_size // 2, y_center - patch_size // 2
            crop_x = int(min(max(crop_x, 0), W - patch_size - 1))
            crop_y = int(min(max(crop_y, 0), H - patch_size - 1))
            patch_img = slide.read(location=(crop_x, crop_y), size=(patch_size, patch_size), scale=scale)
            col = cur_roi_indx % num_col
            row = cur_roi_indx // num_col

            start_x, start_y = (col + 1) * border_thickness + col * patch_size, (
                    row + 1) * border_thickness + row * patch_size

            xmin, ymin = contour_coords[0]
            xmax, ymax = contour_coords[2]
            xmin = int((xmin - crop_x) / scale + start_x)
            xmax = int((xmax - crop_x) / scale + start_x)
            ymin = int((ymin - crop_y) / scale + start_y)
            ymax = int((ymax - crop_y) / scale + start_y)
            x_coords = [xmin, xmax, xmax, xmin]
            y_coords = [ymin, ymin, ymax, ymax]

            x_patch_coords = [start_x, start_x + patch_size, start_x + patch_size, start_x]
            y_patch_coords = [start_y, start_y, start_y + patch_size, start_y + patch_size]

            combined_image[start_y:start_y + patch_size, start_x:start_x + patch_size, :] = patch_img
            patch_cells.append(
                {
                    "id": str(int(random.random() * 100000)),
                    "path": {"x": x_coords, "y": y_coords},
                    "remark": label,
                    "patch_path": {"x": x_patch_coords, "y": y_patch_coords},
                    "microorganism": False,
                    "positive": False
                })

            cur_roi_indx += 1
            cur_cell_idx += 1
        except:
            cur_cell_idx += 1
    slide_path = slide.filename

    if is_save_label:
        label_path = os.path.join(os.path.dirname(slide_path), 'label.png')
        slide.saveLabel(label_path)
        if os.path.exists(label_path):
            slide_label = Image.open(label_path)
            if (slide_label.mode != 'RGB'):
                slide_label = slide_label.convert("RGB")

            label_width, label_height = slide_label.size
            if label_height > label_width:
                slide_label = slide_label.rotate(90)
                label_width, label_height = label_height, label_width

            resize_ratio = label_size / label_height
            slide_label = slide_label.resize((round(label_width * resize_ratio), label_size))
            slide_label = np.array(slide_label, dtype=np.uint8)
            label_resize_h, label_resize_w = slide_label.shape[0:2]
            upper_label = np.ones((label_resize_h, w, 3), dtype=np.uint8) * 255
            upper_label[0:label_resize_h, 0:label_resize_w, :] = slide_label
            combined_image = np.vstack((combined_image, upper_label))
        combined_image = combined_image.astype(np.uint8)
    os.makedirs(os.path.join(os.path.split(slide_path)[0], 'ai', alg_type), exist_ok=True)
    imageio.imsave(
        os.path.join(os.path.split(slide_path)[0], 'ai', alg_type, 'rois.jpg'),
        combined_image)
    # TODO decode error
    # imageio.imsave(os.path.join(res_path.decode(), 'rois.jpg'), combined_image)
    # algor_type = os.path.splitext(os.path.basename(__file__))[0]
    result_dict[alg_type] = {'diagnosis': result['diagnosis'], 'result': patch_cells}
    result_dict['mpp'] = slide_mpp

    with open(os.path.join(os.path.split(slide_path)[0], 'ai', alg_type, 'rois.json'), 'w', encoding='utf-8') as f:
        json.dump(result_dict, f)


def save_empty_roi(slide_path, alg_type='lct'):
    empty_img = np.zeros((512, 512, 3), dtype=np.uint8)
    imageio.imsave(os.path.join(os.path.split(slide_path)[0], 'ai', alg_type, 'rois.jpg'),empty_img)
    with open(os.path.join(os.path.split(slide_path)[0], 'ai', alg_type, 'rois.json'), 'w', encoding='utf-8') as f:
        json.dump({'diagnosis': '', 'result': []}, f)




def walk_dir(data_dir, file_types= ['.kfb', '.tif', '.svs', '.ndpi', '.mrxs', '.hdx', '.sdpc', '.mds', '.mdsx']):
    # file_types = ['.txt', '.kfb']
    path_list = []
    for dirpath, dirnames, files in os.walk(data_dir):
        for f in files:
            for this_type in file_types:

                if f.lower().endswith(this_type):
                    path_list.append(os.path.join(dirpath, f))
                    break
    return path_list