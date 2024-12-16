#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2021/6/27 23:20
# @Author  : Can Cui
# @File    : db_sql.py
# @Software: PyCharm
# @Comment:


sql_create_tables = '''
    CREATE TABLE  IF NOT EXISTS "Mark_{}" (
    "id" INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    "position" TEXT,
    "method" TEXT,
    "isExport" INTEGER,
    "remark" TEXT,
    "aiResult" TEXT,
    "editable" INTEGER,
    "strokeColor" TEXT,
    "fillColor" TEXT,
    "markType" INTEGER,
    "diagnosis" INTEGER,
    "radius" REAL,
    "createTime" REAL,
    "groupId" INTEGER,
    "areaId" INTEGER,
    "dashed" INTEGER
    );

    CREATE TABLE IF NOT EXISTS  "MarkToTile_{}" (
    "id" INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    "markId" INTEGER,
    "tileId" INTEGER
    );
    CREATE INDEX IF NOT EXISTS  "t_index" ON "MarkToTile_{}" ("tileId");

'''


sql_creat_talbel_Pdl1sCount = '''
    CREATE TABLE IF NOT EXISTS  "Pdl1sCount" (
    "tileId" INTEGER PRIMARY KEY NOT NULL,
    "posTumor" INTEGER,
    "negTumor" INTEGER,
    "posNorm" INTEGER,
    "negNorm" INTEGER
    );
'''

sql_insert_pdl1scount = '''
    INSERT INTO Pdl1sCount("tileId", "posTumor", "negTumor", "posNorm", "negNorm") VALUES (?,?,?,?,?)
    
'''

sql_insert_mark = "INSERT OR IGNORE INTO {}(" \
                  "id," \
                  "position," \
                  "method, " \
                  "isExport, " \
                  "remark, " \
                  "aiResult, " \
                  "editable, " \
                  "strokeColor, " \
                  "fillColor," \
                  "markType, " \
                  "diagnosis, " \
                  "radius, " \
                  "createTime, " \
                  "groupId, " \
                  "areaId, " \
                  "dashed " \
                  ") VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"

sql_insert_mark_no_id = "INSERT OR IGNORE INTO {}(" \
                          "position," \
                          "method, " \
                          "isExport, " \
                          "remark, " \
                          "aiResult, " \
                          "editable, " \
                          "strokeColor, " \
                          "fillColor," \
                          "markType, " \
                          "diagnosis, " \
                          "radius, " \
                          "createTime, " \
                          "groupId, " \
                          "areaId, " \
                          "dashed " \
                          ") VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"

sql_insert_mark_to_tile = '''INSERT INTO {}("markId","tileId") VALUES (?,?)'''

sql_query_ai_id = "select id from ai where ai_name=?"


sql_query_template_id = "select id from Template where aiId=?"

sql_query_groups = "select groupName, id from MarkGroup where templateId=?"
