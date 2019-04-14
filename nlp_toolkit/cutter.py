# encoding: utf-8
"""
@author: leo
@version: 1.0
@license: Apache Licence
@file: cutter.py
@time: 2017/11/19 下午4:25

"""
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import jieba
from jieba import posseg
import logging
from bf_nlp.utils.num_helper import NumDict
from bf_nlp.utils.num_helper import NumberHelper
import pymysql


class SentenceCutter(object):
    """
    句子分词和 Tokenize
    """

    def __init__(self, dicts=list()):
        self.jieba = jieba

        # 词典自定义添加了特定词、用户词典、数字词典
        self.jieba.add_word("成龙", 1000, 'movie_person')
        self.jieba.add_word("快进", 1000)
        for k in ['#UNK', '#PAD', '#BOS', '#EOS']:
            self.jieba.add_word(k, 1000)
        self.jieba.add_word("快进", 1000)
        self.jieba.add_word("快进", 1000)
        self.jieba.add_word("快进", 1000)
        self.jieba.add_word("快进", 1000)
        self.jieba.add_word("人民的名义")
        self.jieba.add_word("上一页")
        self.jieba.add_word("下一页")

        self.num_dic = NumDict()

        self.num_dic, self.num_array = self.num_dic.get_num_data()

        for d in dicts:
            self.jieba.load_userdict(d)

        logging.debug('Jieba load user dicts: {}'.format(dicts))

        data = list()
        # 数字
        data.append(['上一集', '下一集'])
        data.append(self.num_array)

        for f in data:
            for l in f:
                jieba.add_word(l)

        # 加载用户自定义词典
        db_conf = {
            'host': '192.168.11.122',
            'port': 3306,
            'password': 'Wjckvyfshb#9gI1',
            'user': 'aida'
        }
        self.db = pymysql.connect(**db_conf,
                                  charset='utf8mb4',
                                  cursorclass=pymysql.cursors.DictCursor)
        # self.db = conf.get_mysql('ailab')
        self.seg = jieba.Tokenizer()
        self.load_userdict()
        # 词典自定义添加了特定词、用户词典、数字词典
        self.seg.add_word("成龙", 1000, 'movie_person')
        self.seg.add_word("快进", 1000)
        for k in ['#UNK', '#PAD', '#BOS', '#EOS']:
            self.seg.add_word(k, 1000)
        self.seg.add_word("快进", 1000)
        self.seg.add_word("快进", 1000)
        self.seg.add_word("快进", 1000)
        self.seg.add_word("快进", 1000)
        self.seg.add_word("人民的名义")
        self.seg.add_word("上一页")
        self.seg.add_word("下一页")
        data = list()
        # 数字
        data.append(['上一集', '下一集'])
        data.append(self.num_array)

        for f in data:
            for l in f:
                self.seg.add_word(l)

        self.pos = posseg.POSTokenizer(self.seg)

    def load_userdict(self):
        logging.info('cutter userdict starts to load......')
        try:
            self.db.connect()
            with self.db.cursor() as cursor:
                sql = "select i.entity, i.synonym, l.name from baofeng_ai.ailab_aida_entity_items as i " \
                      "left join baofeng_ai.ailab_aida_entity_list as l on i.entity_id=l.id where i.isdel = 0 and i.entity_id in (1, 2, 3, 4, 5, 6, 7, 8, 15, 16, 17, 20, 25, 26)"
                # sql = 'select * from baofeng_ai.ailab_aida_entity_items'
                cursor.execute(sql)
                data = cursor.fetchall()
                for item in data:
                    self.seg.add_word(item['entity'], 1000, item['name'])
                    if not item['synonym']:
                        continue
                    synonym_list = item['synonym'].split('|<=|=>|')
                    for synonym in synonym_list:
                        self.seg.add_word(synonym, 1000, item['name'])
            logging.info('cutter load_userdict succeed!')
        except Exception as e:
            logging.error('cutter load_userdict failed, {}'.format(str(e)))
        finally:
            self.db.close()

    def cut(self, sentence, debug=False):
        li = list(jieba.cut(sentence))
        new = []
        for i in li:
            new += self.num_dic.get(i, [i])
        return [word for word in new if word.strip() != '']

    def poscut(self, sentence, debug=False):
        result = list(self.pos.cut(sentence))
        pos = []
        for word, tag in result:
            pos.append(word + '/' + tag)
        return pos

    def tokenize(self, sentence):
        li = list(jieba.cut(sentence))
        new = []

        for i in li:
            n = self.num_dic.get(i, [i])
            new += n
        start = 0
        result = []
        for i in new:
            result.append((i, start))
            start += len(i)
        return result


class CorpusCutter(object):
    """
    文件分词器，只能加载文件dict, 如果想加载库里的实体词需编码，参考aida tokenizer reload
    """

    def __init__(self, **conf):
        dicts = conf.get('dicts', list())
        self.seperator = conf.get('seperator', ' ')
        self.cutter = SentenceCutter(dicts)

    def handle(self, inp, outp=''):
        finput = open(inp, encoding='utf-8')
        foutput = open(outp, 'w', encoding='utf-8')
        logging.info("start handle file {}".format(finput))
        i = 0
        for line in finput:
            line_seg = self.cutter.cut(line)
            foutput.write(self.seperator.join(line_seg) + '\n')
            i += 1
            if i % 1000 == 0:
                logging.info("Saved " + str(i) + " articles/line seg")

        finput.close()
        foutput.close()
        logging.info("Finished Saved " + str(i) + " articles")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    cutter = SentenceCutter()
    text = '我想看速度与激情'
    print(list(cutter.poscut(text)))

    # dicts = ['/home/leo/corpus/2017-11-03-jieba.csv']
    # c = CorpusCutter(dicts=dicts)
    # for file in ['zh_wiki_00', 'zh_wiki_01', 'zh_wiki_02']:
    #     infile = '/home/leo/corpus/wiki_corpus/zh_regulated/std_' + file
    #     outfile = '/home/leo/corpus/wiki_corpus/zh_segmented/seg_' + file
    #     c.handle(infile, outfile)

    # cutter = SentenceCutter(dicts=['/Users/leo/Desktop/ailab/rasa-related/rasa-demo/movie_bot/data/jieba.csv'])
    # cutter = SentenceCutter()
    # sample = [
    #     '播放一八八七年的评分八分的电影快进十分钟八分播放下一期暂停一下第五集',
    #     '播放一下二零零五年的十万个冷笑话第一季二十五分钟',
    #     '播放速度与激情二',
    #     '播放速度与激情2',
    #     '暂停一下',
    #     '我想看人民的名义',
    #     "快进到10分钟",
    #     "hello world ",
    #     "25届金鸡百花电影节暨第33"
    # ]
    # for s in sample:
    #     r = cutter.cut(s, debug=True)
    #     print('-' * 40)
    #     print(s, ' -> ', list(r))
    #     print('=' * 40)
    #


    # a = jieba.Tokenizer()
    # b = jieba.Tokenizer()
    # a.add_word(word='速度与激情', freq=1000, tag = 'movie_name')
    # text = '我想看速度与激情'
    # print(list(a.cut(text)))
    # print(list(b.cut(text)))
    #
    # pos_dt = posseg.POSTokenizer(a)
    # print(list(pos_dt.cut(text)))




