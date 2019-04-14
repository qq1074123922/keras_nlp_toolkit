# encoding: utf-8
"""
@author: leo 
@version: 1.0
@license: Apache Licence
@file: Sentences.py
@time: 2017/11/18 下午3:14

"""
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import json
import random

from nlp_toolkit.cutter import SentenceCutter
from nlp_toolkit.resource.base_data import BaseData

UNTAGGED_INTENT = 'O'
BEGAN_OF_SENTENCE = '#BOS'
END_OF_SENTENCE = '#EOS'
UNKNOWN_WORDS = '#UNK'


def is_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


class Sentence(object):
    def __init__(self,
                 pattern,
                 intent,
                 sentence_tag,
                 sentence_seg):
        self.id = 0
        self.pattern = pattern
        self.intent = intent
        self.sentence = ''.join(sentence_seg)
        self.sentence_tag = sentence_tag
        self.sentence_sen = sentence_seg

    def rasa_json(self):
        """
        {
        "text": "人民的名义有么",
        "intent": "movie.search",
        "entities": [
            {
                "start": 0,
                "end": 5,
                "value": "人民的名义",
                "entity": "movie_name"
             }
          ]
        },
        """
        data = {
            'text': self.sentence,
            'intent': self.intent,
            'entities': []
        }


        start = 0
        for index, word in enumerate(self.sentence_sen):
            key = self.sentence_tag[index]

            if key != UNTAGGED_INTENT:
                entitie = {
                    'start': start,
                    'end': start + len(word),
                    'value': word,
                    'entity': key
                }
                data['entities'].append(entitie)
            start += len(word)

        return data

    def slu_data(self):
        sen = [BEGAN_OF_SENTENCE] + self.sentence_sen + [END_OF_SENTENCE]
        tag = ['O'] + self.sentence_tag + [self.intent]
        return (sen, tag)

    @staticmethod
    def get_sentences(domain_id,count=100,
                      noice_count=100,
                      random_keys=[],
                      random_rate=0.95,
                      cutter=None,
                      db_conf=None):
        base_data = BaseData(db_conf)
        base_data.prepare_data(domain_id)
        data = []
        for _ in range(count):
            patt_tuple, intent_id = random.choice(base_data.sentence_list)
            intent = base_data.intents.get(intent_id,None)
            if intent is None:
                continue
            pattern_id, business_id, pattern = patt_tuple

            b_id, b_tag = base_data.business.get(business_id, (0, ''))
            if b_tag != '':
                intent = intent + '.' + b_tag

            pattern = pattern.split(',')
            sentence_seg = []
            sentence_tag = []
            for i in pattern:
                if is_int(i):
                    key = int(i)
                    if key < 0:
                        prefix = 'DENY:'
                        key = abs(key)
                    else:
                        prefix = ''
                    slot_id, slot = base_data.slots.get(key,(None,None))
                    if slot_id is None:
                        continue
                    if random.random() > random_rate and slot in random_keys:
                        sentence_seg.append(UNKNOWN_WORDS)
                    else:
                        slot_word = base_data.slot_words.get(key,None)
                        if slot_word is None:
                            continue
                        word_id, word = random.choice(slot_word)
                        sentence_seg.append(word)
                    sentence_tag.append(prefix + slot)
                else:
                    sentence_seg.append(i)
                    sentence_tag.append('O')
            sen = Sentence(pattern, intent, sentence_tag, sentence_seg)
            sen.id = pattern_id
            data.append(sen)

        if cutter is None:
            cutter = SentenceCutter()

        for _ in range(noice_count):
            id, sentence = random.choice(base_data.noise_list)
            intent = 'chit.chat'
            sentence_sen = cutter.cut(sentence)
            sentence_tag = ['O'] * len(sentence_sen)
            sen = Sentence(sentence_sen,
                           intent,
                           sentence_tag,
                           sentence_sen)
            sen.id = id
            data.append(sen)

        return data


if __name__ == "__main__":
    print("Hello world")
    data = Sentence.get_sentences(count=10000, noice_count=10)
    common_examples = []
    for i in data:
        try:
            json.dumps(i.rasa_json(), ensure_ascii=False, indent=2)
            common_examples.append(i.rasa_json())
        except:
            print(i.rasa_json())
            pass
    data = {
        "rasa_nlu_data": {
            "common_examples": common_examples
        }
    }
    with open('output.json', 'wb') as f:
        f.write(json.dumps(data, ensure_ascii=False, indent=2).encode("UTF-8"))

    # for d in data:
    #     print()
    #     print(d.id)
    #     # print(d.pattern)
    #     # print(d.intent)
    #     # print(d.sentence)
    #     # print(d.sentence_tag)
    #     # print(d.sentence_sen)
    #     print('-----------')
