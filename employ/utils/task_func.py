#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2021/8/11 15:12
# @Author  : Can Cui
# @File    : utils.py
# @Software: PyCharm
# @Comment:
import os, sys
import json
import requests
from shutil import copyfile, copytree, move, rmtree

proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, proj_root)


# def task_down(slide_path, ai_type, res=None):
#
#     if res is None:
#         res = {"done":1}
#
#     res_path = os.path.join(os.path.split(slide_path)[0], 'ai', ai_type)
#     res_path = res_path.encode('gbk')
#     os.makedirs(res_path, exist_ok=True)
#     with open(os.path.join(res_path, 'done.txt'.encode('gbk')), 'w', encoding='utf-8') as f:
#         pass
#
#     with open(os.path.join(res_path, 'done.json'.encode('gbk')), 'w', encoding='utf-8') as f:
#         json.dump(res, f)
#
# def inform(response_data):
#     return  requests.post(url=back_addr + '/aipath/api/ai/inform', data=json.dumps(response_data))


def save_prob(slide_path, result, ai_type):
    try:
        if 'slide_pos_prob' in result:
            slide_pos_prob = result['slide_pos_prob'][0]
            slide_diagnosis = result['diagnosis']
            tbs_label = result['tbs_label']

            save_dict ={
            'slide_path': slide_path,
            'filename': os.path.basename(slide_path),
            'NILM': round(float(slide_pos_prob[0]), 5),
            'ASC-US': round(float(slide_pos_prob[1]),5),
            'LSIL': round(float(slide_pos_prob[2]),5),
            'ASC-H': round(float(slide_pos_prob[3]),5),
            'HSIL': round(float(slide_pos_prob[4]),5),
            'AGC':round(float(slide_pos_prob[5]),5),
            'diagnosis': slide_diagnosis,
            'tbs_label': tbs_label
            }

            res_path = os.path.join(os.path.split(slide_path)[0])
            os.makedirs(res_path, exist_ok=True)

            with open(os.path.join(res_path, 'prob_{}.json'.format(ai_type.__name__)), 'w', encoding='utf-8') as f:
                json.dump(save_dict, f)

            return_dict = {k:save_dict[k] for k in ['NILM', 'ASC-US', 'LSIL', 'ASC-H', 'HSIL', 'AGC']}
            return return_dict

    except Exception as e:
        print(e)
    return False



def walk_dir(data_dir, file_types= ['.kfb', '.tif', '.svs', '.ndpi', '.mrxs', '.hdx']):
    # file_types = ['.txt', '.kfb']
    path_list = []
    for dirpath, dirnames, files in os.walk(data_dir):
        for f in files:
            for this_type in file_types:

                if f.lower().endswith(this_type):
                    path_list.append(os.path.join(dirpath, f))
                    break
    return path_list

def copy_slide(src, dest):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    copyfile(src, dest)
    if os.path.exists(os.path.splitext(src)[0]):
        copytree(os.path.splitext(src)[0], os.path.splitext(dest)[0])

def remove_slide(src):
    if src is not None and os.path.exists(src):
        os.remove(src)
        if os.path.exists(os.path.splitext(src)[0]):
            rmtree(os.path.splitext(src)[0])

def move_slide(src, dest):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    move(src, dest)
    if os.path.exists(os.path.splitext(src)[0]):
        move(os.path.splitext(src)[0], os.path.splitext(dest)[0])