# encoding: utf-8
"""
@author: leo 
@version: 1.0
@license: Apache Licence
@file: base_model.py
@time: 2017/11/19 下午3:26

"""
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import json


class BaseModel(object):
    """
    基础模型，需要有 id 和 _table_name_ 这个属性
    """
    def __init__(self):
        self.__data__ = {
            'id': -1,
        }
        self.__table_name__ = '__table_name__'

    def __repr__(self):
        return "<Model {}>".format(self.dic)

    @property
    def dic(self):
        """
        model 字典
        """
        return self.__data__

    @property
    def json(self):
        return json.dumps(self.__data__, ensure_ascii=False, indent=2)

    def update_with_json_string(self, json_str):
        """
        使用 json 字符串更新模型
        :param json_str: json 字符串
        """
        data = json.loads(json_str, encoding='utf-8')
        self.update_with_dic(data)
        return self

    def update_with_dic(self, new_data):
        """
        使用字典更新模型
        :param new_data: 数据字典
        """
        self.__data__.update(new_data)
        return self

    def sql_update_statement(self, tablename=None, ignore_none=True):
        """
        获取更新 sql 语句
        :param tablename: 表名
        :param ignore_none: 是否忽略None
        """
        if tablename is None:
            tablename = self.__table_name__
        sql = "UPDATE {} SET ".format(tablename)
        set_data = []
        for k, v in self.__data__.items():
            if ignore_none:
                if v is not None and v != "":
                    set_data.append('{}={}'.format(k,v))
            else:
                set_data.append('{}={}'.format(k, v))
        sql += ', '.join(set_data)
        sql += ' WHERE id = {}'.format(self.__data__.get('id', None))
        return sql

    def sql_insert_statement(self, tablename=None):
        """
        获取插入 sql 语句
        :param tablename: 表名
        """
        if tablename is None:
            tablename = self.__table_name__
        sql = "INSERT INTO {} ".format(tablename)
        keys = [str(k) for k, v in self.__data__.items()]
        values = [str(v) for k, v in self.__data__.items()]
        sql += "({}) VALUES ({})".format(', '.join(keys), ', '.join(values))
        return sql


if __name__ == "__main__":
    print("Hello world")

    print("Hello world")

    class History(BaseModel):
        def __init__(self):
            BaseModel.__init__(self)
            self.__data__ = {
                'id': -1,
                'name': ''
            }
            self.__table_name__ = 'new_data'

    m = History()
    print(m.dic)
    m.update_with_dic({'id': 1, 'name': 'hello', 'aaa': '123123'})
    print(m.dic)
    print(m.sql_insert_statement())
