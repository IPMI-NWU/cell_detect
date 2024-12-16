#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2021/6/27 21:58
# @Author  : Can Cui
# @File    : Mark.py
# @Software: PyCharm
# @Comment:

class Mark(object):
    def __init__(self, id=None, path='', method='spot', image=0, aiResult='', remark='', editable=1, strokeColor='',
                 fillColor='', dashed=0, mark_type=2, diagnosis=None, radius=1, creat_time='', group_id=0, area_id=0):
        self.id = id
        self.position = path
        self.method = method
        self.isExport = image
        self.remark = remark
        self.aiResult = aiResult
        self.editable = editable
        self.strokeColor = strokeColor
        self.fillColor = fillColor
        self.markType = mark_type
        self.diagnosis = diagnosis
        self.radius = radius
        self.createTime = creat_time
        self.groupId = group_id
        self.areaId = area_id
        self.dashed = dashed



    def val(self):
        val = (self.position, self.method, self.isExport, self.remark, self.aiResult, self.editable, self.strokeColor,
                    self.fillColor,  self.markType,  self.diagnosis, self.radius,
                    self.createTime, self.groupId, self.areaId, self.dashed,)
        if self.id is not None:
            val  = (self.id,) + val
        return val


class MarkToTile(object):
    def __init__(self, mark_id, tile_id):
        self.mark_id  = mark_id
        self.tile_id = tile_id

        self.insert_sql = '''INSERT INTO MarkToTile(mark_id, tile_id) VALUES(?, ?)'''

    def val(self):
        return (self.mark_id, self.tile_id)

class Pdl1sCount(object):
    def __init__(self, tile_id, pos_tumor=0, neg_tumor=0, pos_norm=0, neg_norm=0):
        self.tile_id = int(tile_id)
        self.pos_tumor = int(pos_tumor)
        self.neg_tumor = int(neg_tumor)
        self.pos_norm = int(pos_norm)
        self.neg_norm = int(neg_norm)

    def val(self):
        return (self.tile_id, self.pos_tumor, self.neg_tumor, self.pos_norm, self.neg_norm)

    def __add__(self, other):
        return Pdl1sCount(
            tile_id=self.tile_id,
            pos_tumor=self.pos_tumor+other.pos_tumor,
            neg_tumor=self.neg_tumor+other.neg_tumor,
            pos_norm=self.pos_norm+other.pos_norm,
            neg_norm=self.neg_norm+other.neg_norm,
        )

    @property
    def total(self):
        return self.pos_norm+self.pos_tumor+self.neg_norm+self.neg_tumor

    def set_tile_id(self, tile_id):
        self.tile_id = tile_id
        return True

if __name__ == '__main__':
    p1 = Pdl1sCount(10, 1,2,3,4)
    p2 = Pdl1sCount(11, 1,2,3,4)


