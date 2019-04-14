# encoding: utf-8
"""
@author: leo 
@version: 1.0
@license: Apache Licence
@file: BaseData.py
@time: 2017/11/18 下午3:14

"""
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import pymysql


class BaseData(object):
    def __init__(self, db_conf=None):
        if db_conf is None:
            db_conf = {
                'host': 'ailab-mysql',
                'db': 'ailab_db',
                'port': 3307,
                'password': 'Bwof23w<mG',
                'user': 'ailab'
            }
        self.db = pymysql.connect(**db_conf,
                                  charset='utf8mb4',
                                  cursorclass=pymysql.cursors.DictCursor)
        self.slots = None
        self.domains = None
        self.intents = None
        self.slot_words = None
        self.sentence = None
        self.sentence_list = None
        self.noise_list = None
        self.business = None

    def prepare_data(self, domain_id):
        self.domains = self.get_domains(domain_id)
        self.slots = self.get_slots(domain_id)
        self.intents = self.get_intents(domain_id)
        slot_ids = [str(k) for k, v in self.slots.items()]
        self.slot_words = self.get_slot_words(slot_ids)
        self.sentence = self.get_sentence_list(domain_id)
        self.business = self.get_business(domain_id)

        self.sentence_list = []
        for intent, patterns in self.sentence.items():
            for pat in patterns:
                self.sentence_list.append((pat, intent))
        self.noise_list = self.get_noise_list()

    def get_domains(self, domain_id):
        try:
            self.db.connect()
            with self.db.cursor() as cursor:
                sql = "select `id`, `tag` from ai_domain where isdel = 0 and id = {};".format(domain_id)
                cursor.execute(sql)
                result = cursor.fetchall()
                data = BaseData.format_data(result, 'id', 'tag')
        except:
            data = {}
        finally:
            self.db.close()
        return data

    def get_business(self, domain_id):
        try:
            self.db.connect()
            with self.db.cursor() as cursor:
                sql = "select `id`, `tag` from ai_business where isdel = 0 and domain_id = %s;"
                cursor.execute(sql, domain_id)
                result = cursor.fetchall()
                data = BaseData.format_data(result, 'id', 'tag')
        except:
            data = {}
        finally:
            self.db.close()
        return data

    def get_slot_words(self, slot_ids):
        try:
            self.db.connect()
            with self.db.cursor() as cursor:
                sql = "select `id`, `word`, `slot_id` from ailab_db.ai_slot_word where isdel = 0 and slot_id in({}); ".format(
                    ",".join(slot_ids))
                cursor.execute(sql)
                result = cursor.fetchall()
                data = BaseData.format_data(result,
                                            'slot_id',
                                            'word',
                                            is_list=True)
        except:
            data = {}
        finally:
            self.db.close()
        return data

    def get_slots(self, domain_id):
        try:
            self.db.connect()
            with self.db.cursor() as cursor:
                sql = "select `id`,`tag_in_nlp` from ai_slot where isdel = 0 and domain_id in (0,%s);"
                cursor.execute(sql, [domain_id])
                result = cursor.fetchall()
                data = BaseData.format_data(result,
                                            'id',
                                            'tag_in_nlp')
        except:
            data = {}
        finally:
            self.db.close()
        return data

    def get_intents(self, domain_id):
        try:
            self.db.connect()
            with self.db.cursor() as cursor:
                sql = "select `id`, `tag` , `domain_id` from `ai_intents` where isdel = 0 and domain_id = %s;"
                cursor.execute(sql, [domain_id])
                result = cursor.fetchall()
                data = {}
                for i in result:
                    domain_id, domain = self.domains[i['domain_id']]
                    intent = domain + '.' + i['tag']
                    data[i['id']] = intent
        except Exception as e:
            print('error ', str(e))

            data = {}
        finally:
            self.db.close()
        return data

    def get_sentence_list(self, domain_id):
        try:
            self.db.connect()
            with self.db.cursor() as cursor:
                sql = "SELECT `id`, `intent_id`, `text`, `business_id` FROM ai_sentence where isdel = 0 and domain_id = %s;"
                cursor.execute(sql, domain_id)
                result = cursor.fetchall()

                data = {}
                for item in result:
                    intent = item['intent_id']
                    d = (item['id'], item['business_id'], item['text'])
                    if intent in data:
                        data[intent].append(d)
                    else:
                        data[intent] = [d]

                # data = BaseData.format_data(
                #     result, 'intent_id', 'text', is_list=True)
        except:
            data = {}
        finally:
            self.db.close()
        return data

    def get_noise_list(self):
        try:
            self.db.connect()
            with self.db.cursor() as cursor:
                sql = "SELECT `id`, `text` FROM ai_noise_corpus;"
                cursor.execute(sql)
                result = cursor.fetchall()
                data = [(i['id'], i['text']) for i in result]
        except:
            data = []
        finally:
            self.db.close()
        return data

    @staticmethod
    def format_data(data, id_key, value_key, is_list=False):
        result = {}
        for item in data:
            id = item[id_key]
            value = (item.get('id'), item[value_key])
            if is_list:
                if id in result:
                    result[id].append(value)
                else:
                    result[id] = [value]
            else:
                result[id] = value
        return result


if __name__ == "__main__":
    db = BaseData()
    db.prepare_data(14)

    # print(db.slots)
    # print(db.domains)
    # print(db.intents)
    # print(db.slot_words)
    # print(db.sentence)
    print(db.business)
    # print(db.noise_list)

    # print(db.noise_list[:10])
