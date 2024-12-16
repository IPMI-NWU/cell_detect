#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2020/12/18 10:44
# @Author  : Can Cui
# @File    : db_manager.py
# @Software: PyCharm
# @Comment:
import os
import sqlite3
import traceback

from employ.Slide import openSlide
from shapely.geometry import Polygon, box
import math
import numpy as np
import cv2
from collections import Counter
from .db_sql import *
from .db_object import *
import json
from .id_generator import get_guid


class SliceDBManager(object):

    def __init__(self, db_path, slide_path, ai_type, tile_size=128, max_num_per_tile=5000, clean_mode=False):
        self.conn = sqlite3.connect(db_path, isolation_level="")  # 连接到SQLite数据库
        self.cursor = self.conn.cursor()

        # print(self.get_template_folder_path())
        # self.conn1 = sqlite3.connect(os.path.join(self.get_template_folder_path(), 'dipath.db'), isolation_level="")  # 连接到SQLite数据库
        # self.cursor1 = self.conn1.cursor()
        # self.conn2 = sqlite3.connect(os.path.join(self.get_template_folder_path(), 'slice.db'), isolation_level="")
        # self.cursor2 = self.conn2.cursor()

        self.max_num_per_tile = max_num_per_tile
        self.tile_size = tile_size

        # Load slide information
        self.slide = openSlide(slide_path)
        self.maxlvl = self.slide.maxlvl
        self.height = self.slide.height
        self.width = self.slide.width

        self.mark_table_name = "Mark_label_{}".format(ai_type)
        self.mark_to_tile_table_name = "MarkToTile_label_{}".format(ai_type)
        self.ai_type = ai_type

        if not clean_mode:
            self.pyramid_dict = self.generate_pyramid()
            self.tile2id_dict, self.id2tile_dict = self.generate_tile_id_dict()

        try:
            self.cursor.executescript(sql_create_tables.format(ai_type, ai_type, ai_type))
            self.conn.commit()
        except Exception as e:
            print(e)
            self.conn.rollback()

        # print(self.cursor.execute('SELECT name FROM sqlite_master').fetchall())
        # import pdb; pdb.set_trace()

    def __del__(self):
        self.cursor.close()
        self.conn.close()

    def get_ai_id_by_name(self, ai_type):
        res = self.cursor1.execute(sql_query_ai_id, (ai_type,)).fetchone()
        if res:
            return res[0]
        else:
            raise ValueError("Unrecognized ai_type, {} is not in Ai Table".format(ai_type))

    def execute_sql(self, sql, val=None):
        try:
            if val:
                self.cursor.execute(sql, val)
            else:
                self.cursor.execute(sql)
            self.conn.commit()
        except Exception as e:
            print(e)
            self.conn.rollback()

    def execute_script(self, sql):
        try:
            self.cursor.executescript(sql)
            self.conn.commit()
        except Exception as e:
            print(e)
            self.conn.rollback()

    def build_pyramid_dict(self):
        pyramid_dict = {}
        for z in range(0, self.maxlvl + 1):
            stride = self.tile_size * 2 ** (self.maxlvl - z)
            pyramid_dict[z] = {}
            for x in range(0, self.width, stride):
                for y in range(0, self.height, stride):
                    x_tile, y_tile = x // stride, y // stride
                    pyramid_dict[z][(x_tile, y_tile)] = []
        return pyramid_dict

    def break_tile_recursively(self, all_tile_list, search_tile_list, z, search_max_lvl):
        if z > search_max_lvl:
            return []
        tile_to_break_list = []
        res_tile_list = []
        all_tile_set = set(all_tile_list)
        if z == search_max_lvl:
            return all_tile_list
        else:
            for tile in search_tile_list:
                sub_tile_set = set(self.project_to_z_level(tile, dest_level=search_max_lvl))
                if sub_tile_set.issubset(all_tile_set):
                    res_tile_list.append(tile)
                    all_tile_set -= sub_tile_set
                else:
                    tile_to_break_list += self.project_to_z_level(tile, dest_level=z + 1)
        all_tile_list = list(all_tile_set)
        return self.break_tile_recursively(all_tile_list, tile_to_break_list, z + 1, search_max_lvl) + res_tile_list

    def get_rectangle_region_tile_pyramid(self, x_coords, y_coords):
        '''
        get tiles of a rectangle region, combine tiles inside the rectangle
        :param x_coords:
        :param y_coords:
        :return:
        '''
        search_max_lvl = self.maxlvl - 2
        all_tile_list = self.get_tile_position(x_coords, y_coords, z=search_max_lvl)
        lvl10_tile_list = self.get_tile_position(x_coords, y_coords, z=10)
        tile_list = self.break_tile_recursively(all_tile_list, lvl10_tile_list, z=10, search_max_lvl=search_max_lvl)
        tile_id_list = []
        for tile in tile_list:
            x, y, z = tile
            tile_id_list.append(self.tile2id_dict[x, y, z])
        return tile_id_list

    def get_tile_position(self, x_coords, y_coords, z=None):
        assert len(x_coords) == len(y_coords), 'Error, coordinates length is mismatched.'
        tile_list = []
        num_coords = len(x_coords)
        z = self.maxlvl if z is None else z
        this_level_tile_size = self.tile_size * (2 ** (self.maxlvl - z))

        tile_xmax, tile_ymax = math.ceil(self.width / this_level_tile_size), math.ceil(
            self.height / this_level_tile_size)
        tile_x_list, tile_y_list = [x for x in range(tile_xmax)], [y for y in range(tile_ymax)]

        if num_coords == 1:  # Point
            x = math.floor(x_coords[0] / this_level_tile_size)
            y = math.floor(y_coords[0] / this_level_tile_size)
            if x in tile_x_list and y in tile_y_list:
                tile_list.append((x, y, z))

        elif num_coords <= 4:  # Triangle or Rectangle
            xmin, xmax = math.floor(min(x_coords) / this_level_tile_size), math.floor(
                max(x_coords) / this_level_tile_size)
            ymin, ymax = math.floor(min(y_coords) / this_level_tile_size), math.floor(
                max(y_coords) / this_level_tile_size)
            for x in range(xmin, xmax + 1):
                for y in range(ymin, ymax + 1):
                    if x in tile_x_list and y in tile_y_list:
                        tile_list.append((x, y, z))
        else:  # Polygon
            merge_coords = list(zip(x_coords, y_coords))
            xmin, xmax = math.floor(min(x_coords) / this_level_tile_size), math.floor(
                max(x_coords) / this_level_tile_size)
            ymin, ymax = math.floor(min(y_coords) / this_level_tile_size), math.floor(
                max(y_coords) / this_level_tile_size)
            poly = Polygon(merge_coords)
            # plt.plot(*poly.exterior.xy)

            for x in range(xmin, xmax + 1):
                for y in range(ymin, ymax + 1):
                    if x in tile_x_list and y in tile_y_list:
                        tile_box = box(x * this_level_tile_size, y * this_level_tile_size,
                                       (x + 1) * this_level_tile_size, (y + 1) * this_level_tile_size)
                        if poly.intersects(tile_box):
                            tile_list.append((x, y, z))
        return tile_list

    def generate_tile_id_dict(self):
        id = 0
        tile2id_dict = {}
        id2tile_dict = {}
        for z in range(self.maxlvl, -1, -1):
            num_col = math.ceil(self.width / (self.tile_size * 2 ** (self.maxlvl - z)))
            num_row = math.ceil(self.height / (self.tile_size * 2 ** (self.maxlvl - z)))
            for y in range(num_row):
                for x in range(num_col):
                    tile2id_dict[(x, y, z)] = id
                    id2tile_dict[id] = (x, y, z)
                    id += 1
        return tile2id_dict, id2tile_dict

    def tile_to_id(self, x, y, z):
        # (x,y,z) -> id
        id = 0
        for level in range(self.maxlvl, z, -1):
            num_col = math.ceil(self.width / (self.tile_size * 2 ** (self.maxlvl - level)))
            num_row = math.ceil(self.height / (self.tile_size * 2 ** (self.maxlvl - level)))
            id += num_col * num_row
        num_col = math.ceil(self.width / (self.tile_size * 2 ** (self.maxlvl - z)))
        id += (x + y * num_col)
        return id

    def id_to_tile(self, id):
        # id -> (x,y,z)
        z = self.maxlvl
        for level in range(self.maxlvl, -1, -1):
            num_col = math.ceil(self.width / (self.tile_size * 2 ** (self.maxlvl - level)))
            num_row = math.ceil(self.height / (self.tile_size * 2 ** (self.maxlvl - level)))
            if id < num_col * num_row:
                z = level
                break
            else:
                id -= num_col * num_row
        num_col = math.ceil(self.width / (self.tile_size * 2 ** (self.maxlvl - z)))
        y = math.floor(id / num_col)
        x = id - y * num_col
        return (x, y, z)

    def generate_pyramid(self):
        pyramid_dict = {}
        for level in range(self.maxlvl, -1, -1):
            this_level_tile_list = []
            num_col = math.ceil(self.width / (self.tile_size * 2 ** (self.maxlvl - level)))
            num_row = math.ceil(self.height / (self.tile_size * 2 ** (self.maxlvl - level)))
            for y in range(num_row):
                for x in range(num_col):
                    this_level_tile_list.append((x, y))
            pyramid_dict[level] = this_level_tile_list
        return pyramid_dict

    def get_rectangle_mark_to_tile_insert_list(self, mark_id, tile_list):
        insert_list = []
        for z in range(9, self.maxlvl + 1):
            for tile in tile_list:
                maxlvl_x, maxlvl_y, _ = tile
                this_level_tile_x, this_level_tile_y = math.floor(maxlvl_x / (2 ** (self.maxlvl - z))), math.floor(
                    maxlvl_y / (2 ** (self.maxlvl - z)))
                tile_id = self.tile2id_dict[this_level_tile_x, this_level_tile_y, z]
                insert_list.append(MarkToTile(mark_id=mark_id, tile_id=tile_id).val())
        insert_list = list(set(insert_list))
        return insert_list

    def insert_records_unlimited(self, mark_id, tile_list):
        insert_list = []
        for z in range(9, self.maxlvl + 1):
            for tile in tile_list:
                maxlvl_x, maxlvl_y, _ = tile
                this_level_tile_x, this_level_tile_y = math.floor(maxlvl_x / (2 ** (self.maxlvl - z))), math.floor(
                    maxlvl_y / (2 ** (self.maxlvl - z)))
                tile_id = self.tile2id_dict[this_level_tile_x, this_level_tile_y, z]
                insert_list.append(MarkToTile(mark_id=mark_id, tile_id=tile_id).val())
        insert_list = list(set(insert_list))
        try:
            self.cursor.executemany(sql_insert_mark_to_tile.format(self.mark_to_tile_table_name), insert_list)
            self.conn.commit()
        except Exception as e:
            print(e)
            self.conn.rollback()

    def get_last_mark_id(self):
        cursor1 = self.conn.cursor()
        mark = Mark(id=None)
        cursor1.execute(sql_insert_mark_no_id.format(self.mark_table_name), mark.val())
        last_row_id = cursor1.lastrowid - 1
        self.conn.rollback()
        return last_row_id

    def project_to_z_level(self, source_tile=(0, 0, 0), dest_level=0):
        x, y, z = source_tile
        dest_tile_list = []
        # assert (x,y) in self.pyramid_dict[z], 'Wrong tile position'
        if (x, y) not in self.pyramid_dict[z]:
            return dest_tile_list

        xmin, xmax = math.floor(x * 2 ** (dest_level - z)), math.ceil((x + 1) * 2 ** (dest_level - z))
        ymin, ymax = math.floor(y * 2 ** (dest_level - z)), math.ceil((y + 1) * 2 ** (dest_level - z))

        for col in range(xmin, xmax):
            for row in range(ymin, ymax):
                dest_tile_list.append((col, row, dest_level))
        return dest_tile_list

    def get_MarkID_tileID_point_with_count(self, center_coords_np, cell_types_np, mark_ids_np, downsample_ratio=4):
        '''
        store cell count in db
        :param center_coords_np:
        :param cell_types_np:
        :param mark_ids_np:
        :return:
        '''
        # np.save('a.npy', center_coords_np)
        # np.save('b.npy', cell_types_np)
        # np.save('c.npy', mark_ids_np)
        cell_count_in_tile_list = []
        maxlvl_store_count = self.maxlvl - 2
        import time
        tic = time.time()
        # maxlvl store marks
        scaled_tile_np = np.floor(center_coords_np / self.tile_size).astype(np.int)
        scaled_tile_list = list(scaled_tile_np)
        tile_id_list = [self.tile2id_dict[(x, y, self.maxlvl)] for (x, y) in scaled_tile_list]
        mark_id_list = list(map(int, mark_ids_np))
        mark_to_tile_list = list(zip(mark_id_list, tile_id_list))

        pos_tumor_idx = np.where(cell_types_np == 3)[0]
        pos_tumor_coords = center_coords_np[pos_tumor_idx]
        pos_tumor_ids = mark_ids_np[pos_tumor_idx]
        upper_lvl_coords_np = pos_tumor_coords
        upper_lvl_mark_ids_np = pos_tumor_ids
        for z in range(max(self.maxlvl - 1, 9), 9, -1):
            # this_lvl_downsample_ratio = downsample_ratio ** ((self.maxlvl-z))
            if upper_lvl_coords_np.shape[0] > 4000 and z < self.maxlvl - 1:
                this_lvl_center_coords_np = upper_lvl_coords_np[::downsample_ratio, :]
                this_lvl_mark_ids_np = upper_lvl_mark_ids_np[::downsample_ratio]

                upper_lvl_coords_np = this_lvl_center_coords_np
                upper_lvl_mark_ids_np = this_lvl_mark_ids_np
            else:
                this_lvl_center_coords_np = upper_lvl_coords_np
                this_lvl_mark_ids_np = upper_lvl_mark_ids_np
            this_lvl_scaled_tile_np = np.floor(
                this_lvl_center_coords_np / (self.tile_size * 2 ** (self.maxlvl - z))).astype(np.int)
            this_lvl_scaled_tile_list = list(this_lvl_scaled_tile_np)
            tile_id_list = [self.tile2id_dict[(x, y, z)] for (x, y) in this_lvl_scaled_tile_list]
            mark_id_list = list(map(int, this_lvl_mark_ids_np))
            mark_to_tile_list += list(zip(mark_id_list, tile_id_list))

        # #the rest lvl store cell count
        tic = time.time()
        temp_dict = {}

        for z in range(max(maxlvl_store_count, 9), 9, -1):
            scaled_tile_np = np.floor(center_coords_np / (self.tile_size * 2 ** (self.maxlvl - z))).astype(np.int)
            scaled_tile_list = list(scaled_tile_np)
            tile_id_list = [self.tile2id_dict[(x, y, z)] for (x, y) in scaled_tile_list]
            for idx, tile_id in enumerate(tile_id_list):
                if tile_id in temp_dict:
                    temp_dict[tile_id] += Counter([cell_types_np[idx]])
                else:
                    temp_dict[tile_id] = Counter([cell_types_np[idx]])

        for tile_id, cell_count in temp_dict.items():
            pdl1_count = Pdl1sCount(
                tile_id=tile_id,
                pos_tumor=cell_count[3],
                neg_tumor=cell_count[1],
                pos_norm=cell_count[2],
                neg_norm=cell_count[0])
            cell_count_in_tile_list.append(pdl1_count.val())

        return mark_to_tile_list, cell_count_in_tile_list

    def get_markID_tileID_point(self, center_coords_np, mark_ids_np, max_num_per_tile=40):
        '''
        correlate marks with tile
        :param center_coords_np:
        :param mark_ids_np:
        :param max_num_per_tile: max num of marks stored per tile
        :return:
        '''
        mark_to_tile_list = []
        temp_dict = {}

        # maxlvl
        scaled_tile_np = np.floor(center_coords_np / self.tile_size).astype(np.int)
        scaled_tile_list = list(scaled_tile_np)
        tile_id_list = [self.tile2id_dict[(x, y, self.maxlvl)] for (x, y) in scaled_tile_list if
                        self.tile2id_dict[(x, y, self.maxlvl)]]
        mark_id_list = list(map(int, mark_ids_np))
        mark_to_tile_list += list(zip(mark_id_list, tile_id_list))
        for tile, mark_id in zip(scaled_tile_list, mark_id_list):
            x, y = tile
            if (x, y, self.maxlvl) not in temp_dict:
                temp_dict[(x, y, self.maxlvl)] = [mark_id]
            else:
                temp_dict[(x, y, self.maxlvl)].append(mark_id)

        # the rest lvl
        for z in range(self.maxlvl - 1, 6, -1):
            # import pdb; pdb.set_trace()
            this_lvl_tiled_np = np.floor(center_coords_np / (self.tile_size * 2 ** (self.maxlvl - z))).astype(np.int)

            unique_this_lvl_tiled_np = np.unique(this_lvl_tiled_np, axis=1)
            unique_this_lvl_tiled_list = list(unique_this_lvl_tiled_np)
            # import pdb; pdb.set_trace()
            for tile in unique_this_lvl_tiled_list:
                print(tile)
                x, y = tile
                if (x, y, z) in self.tile2id_dict:
                    tile_id = self.tile2id_dict[(x, y, z)]
                    upper_tile_list = self.project_to_z_level((x, y, z), z + 1)
                    data_list = []

                    for upper_tile in upper_tile_list:
                        if upper_tile in temp_dict:
                            data_list.append(temp_dict[upper_tile])

                    if len(data_list) > 0:
                        this_tile_data = np.concatenate(data_list)
                        num_data = this_tile_data.shape[0]
                        picked_data = np.random.choice(this_tile_data, min(max_num_per_tile, num_data), replace=False)
                        temp_dict[(x, y, z)] = picked_data
                        for mark_id in picked_data:
                            mark_to_tile_list.append((int(mark_id), tile_id))
        return mark_to_tile_list

    def get_markID_tileID_point_downsample(self, center_coords_np, mark_ids_np, downsample_ratio=4):
        '''
        :param center_coords_np:
        :param mark_ids_np:
        :param downsample_ratio:  num of points downsampled on each level
        :return:
        '''
        mark_to_tile_list = []
        upper_lvl_coords_np = center_coords_np
        upper_lvl_mark_ids_np = mark_ids_np
        for z in range(self.maxlvl, 6, -1):
            # this_lvl_downsample_ratio = downsample_ratio ** ((self.maxlvl-z))
            if upper_lvl_coords_np.shape[0] > 4000 and z < self.maxlvl:
                this_lvl_center_coords_np = upper_lvl_coords_np[::downsample_ratio, :]
                this_lvl_mark_ids_np = upper_lvl_mark_ids_np[::downsample_ratio]
                upper_lvl_coords_np = this_lvl_center_coords_np
                upper_lvl_mark_ids_np = this_lvl_mark_ids_np
            else:
                this_lvl_center_coords_np = upper_lvl_coords_np
                this_lvl_mark_ids_np = upper_lvl_mark_ids_np
            this_lvl_scaled_tile_np = np.floor(
                this_lvl_center_coords_np / (self.tile_size * 2 ** (self.maxlvl - z))).astype(np.int)
            this_lvl_scaled_tile_list = list(this_lvl_scaled_tile_np)
            tile_id_list = [self.tile2id_dict[(x, y, z)] for (x, y) in this_lvl_scaled_tile_list if
                            self.tile2id_dict[(x, y, z)]]
            mark_id_list = list(map(int, this_lvl_mark_ids_np))
            mark_to_tile_list += list(zip(mark_id_list, tile_id_list))
            # print(len(mark_id_list))
            # import pdb; pdb.set_trace()
        return mark_to_tile_list

    def get_markID_tileID_point_unlimited(self, mark_ids_np):
        '''
        :param center_coords_np:
        :param mark_ids_np:
        :param downsample_ratio:  num of points downsampled on each level
        :return:
        '''
        mark_to_tile_list = []
        mark_id_list = list(map(int, mark_ids_np))

        for z in range(self.maxlvl, 6, -1):
            # this_lvl_downsample_ratio = downsample_ratio ** ((self.maxlvl-z))
            # this_lvl_scaled_tile_np = np.floor(
            #     this_lvl_center_coords_np / (self.tile_size * 2 ** (self.maxlvl - z))).astype(np.int)

            for mark_id in mark_id_list:
                this_lvl_tile_list = self.pyramid_dict[z]

                for tile in this_lvl_tile_list:
                    x, y = tile
                    tile_id = self.tile2id_dict[(x, y, z)]
                    mark_to_tile_list.append((mark_id, tile_id))
            #
            # this_lvl_scaled_tile_list = list(this_lvl_scaled_tile_np)
            # tile_id_list = [self.tile2id_dict[(x, y, z)] for (x, y) in this_lvl_scaled_tile_list if
            #                 self.tile2id_dict[(x, y, z)]]
            #
            # mark_id_list = list(map(int, this_lvl_mark_ids_np))
            # mark_to_tile_list += list(zip(mark_id_list, tile_id_list))
            # print(len(mark_id_list))
            # import pdb; pdb.set_trace()
        return mark_to_tile_list

    def drawOnImage(self, x_coords, y_coords, tile_list=None):

        xmin, xmax = math.floor(min(x_coords) / self.tile_size), math.floor(max(x_coords) / self.tile_size)
        ymin, ymax = math.floor(min(y_coords) / self.tile_size), math.floor(max(y_coords) / self.tile_size)

        crop_xmin, crop_ymin = xmin * self.tile_size, ymin * self.tile_size
        crop_w, crop_h = (xmax - xmin + 1) * self.tile_size, (ymax - ymin + 1) * self.tile_size
        img_np = self.slide.read((crop_xmin, crop_ymin), (crop_w, crop_h), 1)[:, :, ::-1]
        merge_coords = np.array([[x - crop_xmin, y - crop_ymin] for x, y in zip(x_coords, y_coords)])
        merge_coords = merge_coords.reshape((-1, 1, 2)).astype(int)
        print(merge_coords.shape)
        new_img = cv2.drawContours(img_np, [merge_coords], -1, (255, 0, 0), 10)
        print(len(tile_list))
        for tile_coord in tile_list:
            x, y, z = tile_coord
            x_draw, y_draw = (x - xmin) * self.tile_size, (y - ymin) * self.tile_size
            new_img = cv2.rectangle(new_img, (x_draw, y_draw), (x_draw + self.tile_size, y_draw + self.tile_size),
                                    (0, 0, 255), 3)
            new_img = cv2.putText(new_img, "({},{},{})".format(x, y, z), (x_draw + 200, y_draw + 200),
                                  cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 3, cv2.LINE_AA)
        print("saving...")
        cv2.imwrite("test.png", new_img)

    def update_aiResult(self, id, aiResult, x_coords, y_coords, editable=0):

        if id is None:
            id = get_guid()
            roi_mark = Mark(id=id, path=json.dumps({"x": x_coords, "y": y_coords}), method='rectangle',
                            aiResult=json.dumps(aiResult), mark_type=3, strokeColor='grey', radius=5, image=1,
                            editable=editable)
            self.execute_sql(sql_insert_mark.format(self.mark_table_name), roi_mark.val())
        else:
            self.cursor.execute('''select id from {} where id="{}"'''.format(self.mark_table_name, id))
            if self.cursor.fetchone() is None:
                roi_mark = Mark(id=id, path=json.dumps({"x": x_coords, "y": y_coords}), method='rectangle',
                                aiResult=json.dumps(aiResult), mark_type=3, strokeColor='grey', radius=5, image=1,
                                editable=editable)
                self.execute_sql(sql_insert_mark.format(self.mark_table_name), roi_mark.val())
            else:
                update_ai_area_sql = "UPDATE {} SET aiResult='{}' WHERE id = '{}'".format(self.mark_table_name,
                                                                                          json.dumps(aiResult), id)
                self.execute_sql(update_ai_area_sql)

    def insert_aiResult(self, mark):
        self.execute_sql(sql_insert_mark.format(self.mark_table_name), mark.val())
        pass

    def get_semi_correction_coords(self):
        res = self.cursor.execute(
            '''select position from {} where markType="{}"'''.format(self.mark_table_name, 4)).fetchone()
        if res is not None:
            path = json.loads(res[0])
            semi_correction_coord = (int(path['x'][0]), int(path['y'][0]))
        else:
            semi_correction_coord = (-1, -1)
        return semi_correction_coord

    def insert_point_result(self, np_center_coords, insert_mark_list):
        if np_center_coords.shape[0] > 0:
            mark_insert_id_np = np.arange(np_center_coords.shape[0]) + get_guid()
            markToTile_insert_list = self.get_markID_tileID_point_downsample(np_center_coords, mark_insert_id_np,
                                                                             downsample_ratio=4)
            try:
                insert_mark_list_with_id = [(int(mark_insert_id_np[i]),) + insert_mark_list[i] for i in
                                            range(len(insert_mark_list))]
                self.cursor.executemany(sql_insert_mark.format(self.mark_table_name), insert_mark_list_with_id)
                self.cursor.executemany(sql_insert_mark_to_tile.format(self.mark_to_tile_table_name),
                                        markToTile_insert_list)
                self.conn.commit()
            except Exception as e:
                print(traceback.print_exc())
                self.conn.rollback()

    def insert_point_result_unlimited(self, np_center_coords, insert_mark_list):
        if np_center_coords.shape[0] > 0:
            mark_insert_id_np = np.arange(np_center_coords.shape[0]) + get_guid()
            markToTile_insert_list = self.get_markID_tileID_point_unlimited(mark_insert_id_np)
            try:
                insert_mark_list_with_id = [(int(mark_insert_id_np[i]),) + insert_mark_list[i] for i in
                                            range(len(insert_mark_list))]
                self.cursor.executemany(sql_insert_mark.format(self.mark_table_name), insert_mark_list_with_id)
                self.cursor.executemany(sql_insert_mark_to_tile.format(self.mark_to_tile_table_name),
                                        markToTile_insert_list)
                self.conn.commit()
            except Exception as e:
                print(traceback.print_exc())
                self.conn.rollback()

    def insert_point_result_with_count(self, np_center_coords, np_cell_types, insert_mark_list):

        if np_center_coords.shape[0] > 0:
            mark_insert_id_np = np.arange(np_center_coords.shape[0]) + get_guid()
            insert_mark_list_with_id = [(int(mark_insert_id_np[i]),) + insert_mark_list[i] for i in
                                        range(len(insert_mark_list))]

            self.execute_script(sql_creat_talbel_Pdl1sCount)

            markToTile_insert_list, pdl1s_count_insert_list = self.get_MarkID_tileID_point_with_count(np_center_coords,
                                                                                                      np_cell_types,
                                                                                                      mark_insert_id_np)

            try:
                self.cursor.executemany(sql_insert_mark.format(self.mark_table_name), insert_mark_list_with_id)
                self.cursor.executemany(sql_insert_mark_to_tile.format(self.mark_to_tile_table_name),
                                        markToTile_insert_list)

                self.cursor.executemany(sql_insert_pdl1scount, pdl1s_count_insert_list)
                self.conn.commit()

            except Exception as e:
                print(e)
                self.conn.rollback()


def get_groups(self, ai_name):
    ai_id = self.cursor1.execute(sql_query_ai_id, (ai_name,)).fetchone()[0]
    template_id = self.cursor1.execute(sql_query_template_id, (ai_id,)).fetchone()[0]
    self.cursor1.close()
    self.conn1.close()
    groups = self.cursor2.execute(sql_query_groups, (template_id,)).fetchall()
    group_dict = {}
    for group in groups:
        group_dict[group[0]] = group[1]
    self.cursor2.close()
    self.conn2.close()
    return group_dict


def get_template_folder_path(self):
    template_folder_path = os.path.abspath(r".")
    template_folder_path = template_folder_path.split(os.sep)
    template_folder_path = os.sep.join(template_folder_path)
    template_folder_path = os.path.join(template_folder_path, 'dbTemplate')
    return template_folder_path


if __name__ == '__main__':
    pass

    # db_path = 'db2.db'
    # dbm = SliceDBManager(r"E:\data\dyj\data\2021_06_29_12_47_57_695911\slices\8110613\slice.sqlite",
    #                      slide_path=r"E:\data\dyj\data\2021_06_29_12_47_57_695911\slices\8110613\A085 PD-L1 V+.kfb")
    # tic = time.time()
    # tile_list = dbm.get_rectangle_region_tile_pyramid(x_coords=[6313,19722], y_coords=[19807,34851])
    # print("time = {}".format(time.time()-tic))
    # print(len(list(set(tile_list))))

    # res = dbm.cursor.execute("SELECT id,image,remark,aiResult,path,area_id,mark_type  FROM Mark where ai_id=2 AND mark_type=3 ").fetchall()
    #
    # for item in res:
    #     print(item)
    #     import pdb; pdb.set_trace()

    # tic = time.time()
    # a = Pdl1sCount(tile_id=-1)
    # for i in range(10000):
    #     p = Pdl1sCount(tile_id=i)
    #     a+=p
    # print("time = {}".format(time.time()-tic))
    # a = np.load('a.npy')
    # b = np.load('b.npy')
    # c = np.load('c.npy')
    # dbm.get_MarkID_tileID_point_with_count(a,b,c)

    # import pdb;
    # pdb.set_trace()
