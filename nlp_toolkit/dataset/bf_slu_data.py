# encoding: utf-8
"""
@author: leo 
@version: 1.0
@license: Apache Licence
@file: bf_slu_data.py
@time: 2018/1/26 下午2:45

"""


class SLUData(object):
    def __init__(self, file, count=0):
        self.x_data = []
        self.y_data = []
        self.intent_data = []
        self.data_type = 'train'

        for line in open(file, encoding='utf-8').read().splitlines():
            new_line = line.split('|*$*$*]')
            intent = new_line[0].strip()
            sentence = new_line[1].strip().split(' ')
            tag = new_line[2].strip().split(' ')
            self.x_data.append(sentence)
            self.y_data.append(tag)
            self.intent_data.append(intent)

            if (count > 0) and (len(self.y_data) >= count):
                break


if __name__ == "__main__":
    print("Hello world")
    path = '/Users/leo/Desktop/ailab/ailab/ai_nlp_api/train/t/input/test/slu_corpus.txt'
    s = SLUData(file=path)
    print(s.x_data[:2])
    print(s.y_data[:2])
    print(s.intent_data[:2])
