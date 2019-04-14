# encoding: utf-8
"""
@author: leo
@version: 1.0
@license: Apache Licence
@file: slu_predictor.py
@time: 2017/11/19 下午4:59

"""
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import re
import os
import json
import time
import numpy as np

from keras.preprocessing import sequence

from nlp_toolkit.processor.slu_processor import SLUProcessor
from nlp_toolkit.models.slu_model import JointSLUModel
from nlp_toolkit.cutter import SentenceCutter
from nlp_toolkit.models import NLPResult

UNK = "#UNK"
PAD = "#PAD"
EOS = "#EOS"
BOS = "#BOS"


class RegixHandler(object):
    def __init__(self, name, reg_list, intent, slot):
        self.name = name
        self.reg_list = reg_list
        self.intent = intent
        self.slot = slot


class SLUPredictor(object):
    def __init__(self, data_path,
                 model_name='blstm',
                 data_processor=None,
                 debug=False):
        self.data_path = data_path
        if data_processor is None:
            self.data_processor = SLUProcessor(data_path=data_path)
        else:
            self.data_processor = data_processor
        self.data_processor.read_base_info()

        self.base_path = self.data_processor.model_path

        model_config_json = os.path.join(
            self.base_path, 'trained_models', model_name + '_model_config.json')
        with open(model_config_json, 'r', encoding='utf-8') as f:
            text = f.read().replace('\'', '"')
            conf = json.loads(text)

        self.model = JointSLUModel(data_path=data_path, **conf)
        self.model.build()
        self.model.load_weight()

        self.info = conf
        self.debug = debug
        self.cutter = SentenceCutter(self.data_processor.jieba_dicts)

        print("slots2idx is:", self.data_processor.idx2slots)

    def handle_with_model(self, sentence, res_model):
        res_model.handle_method = 'slu_model'
        idx = self.data_processor.line_to_idx(sentence)
        if self.debug:
            print('{} -> {}'.format(sentence, idx))

        idx = sequence.pad_sequences([idx],
                                     maxlen=self.model.time_length,
                                     dtype='int32',
                                     padding='pre')[0]
        tag_result = self.model.model.predict(np.asarray([idx]), 1, 0)
        tag_result_arg = tag_result.argmax(2)[0][-len(sentence):]
        tags = self.data_processor.idx_to_slots(tag_result_arg)
        unknown_words = self.data_processor.get_unkown_words(sentence)
        res_model.unknown_words = unknown_words
        entities = []
        rasa_entities = []

        word_start = 0
        for index in range(len(sentence)):
            if index == 0:
                pass
            elif index == len(sentence) - 1:
                res_model.intent = tags[-1:][0]
            else:
                if tags[index] != 'O':
                    entities.append((tags[index], sentence[index]))
                    value = sentence[index]
                    rasa_entitie = {
                        "entity": tags[index],
                        "start": word_start,
                        "end": word_start + len(value),
                        "value": sentence[index]
                    }
                    rasa_entities.append(rasa_entitie)
                word_start += len(sentence[index])

        res_model.rasa_entities = rasa_entities
        res_model.entities = entities
        return res_model

    def predict(self, query, last_intent=''):
        res_model = NLPResult()
        res_model.query = query
        sentence = self.cutter.cut(res_model.query)
        sentence = sentence[:self.data_processor.padding_size - 2]

        res_model.terms = sentence.copy()
        start = time.time()
        sentence.append(EOS)
        sentence.insert(0, BOS)
        res_model = self.handle_with_model(sentence, res_model)

        res_model.spend_time = time.time() - start
        res_model.handle_method = 'slu_model'
        return res_model


if __name__ == "__main__":
    conf = {}
    data_path = '/Users/leo/Desktop/ailab/ailab/ai_nlp_api/app/data/slu_model/2017.12.06'
    predictor = SLUPredictor(data_path=data_path)

    predictor.debug = True
    targets = [
        ("暂停", ""),
        ("中医基础理论有声读物", ""),
        ("我要听喜马拉雅全球流行音乐金榜", ""),
        ('播放成龙', ""),
        ("有没有郭富城的恐怖片", ""),
        ("岩松信2010年的短片", ""),
        ("我要看英国短片", ""),
        ("不要美国的", ""),
        ("不要美国的", "_movie.query"),
        ("不要美国的", "_music.query"),
        ("我想看人民的名义", ""),
        ("有没有权力的游戏", ""),
        ("权力的游戏", ""),
        ("在吃什么我就叫你过来找你能把我咋样呀dsad", ""),
        ("我想看2011年美国的恐怖片", ""),
        ("播放", "")
    ]

    r = predictor.predict("我想看2011年美国的恐怖片")
    # for k, v in r.entities:
    #     print(k, '++++', v)
    for sen in targets:
        sents, intent = sen
        result = predictor.predict(sents, intent)
        print(json.dumps(result.slu_format(), indent=2, ensure_ascii=False))
