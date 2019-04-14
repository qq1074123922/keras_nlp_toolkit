#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : 2017-12-07 15:10:06
# @Author  : leo 
# @Version : 1.0.0
import argparse
import os
import json
import logging
import numpy as np
import gensim
import multiprocessing
from nlp_toolkit.macros import PAD, UNK, BOS, EOS

MARKED_KEYS = [PAD, UNK, BOS, EOS]


class MySentences(object):
    def __init__(self, files):
        self.files = files

    def __iter__(self):
        """
        用于分行读取数据，训练词向量时候只需要一次读取一行，节省内存资源
        更多的请参照 yield 函数用法
        """
        for file in self.files:
            for line in open(file, encoding='utf-8').read().splitlines():
                if file.endswith('.yml'):
                    line = line.replace('-', '')
                yield line.strip().lower().split()


class Word2Vec(object):
    def __init__(self):
        self.model = gensim.models.Word2Vec()
        self.embedding_matrix = []
        self.embedding_size = 100
        self.word2idx = {}
        self.idx2word = {}
        self.marked_key_dict = {}

    def load_gensim(self, path, count=None):
        model_path = os.path.join(path, 'w2v.model')
        self.word2idx = {}
        self.idx2word = {}
        self.model = gensim.models.Word2Vec.load(model_path)
        info_json_path = os.path.join(path, 'info.json')

        vocab_list = []
        try:
            with open(info_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.marked_key_dict = data.get('MARKED_KEYS', {})
                for word in MARKED_KEYS:
                    vector = [float(i) for i in self.marked_key_dict[word].split(',')]
                    vocab_list.append((word, np.array(vector, dtype=np.float32)))

        except Exception as e:
            vocab_list = []
            logging.error('Read info json path failed with error {}'.format(str(e)))

        if len(vocab_list) == 0:
            for index, key in enumerate(MARKED_KEYS):
                if index != 0:
                    vector = np.random.uniform(-0.25, 0.25, self.embedding_size)
                else:
                    vector = np.zeros(self.embedding_size)
                vocab_list.append((key, vector))
            with open(info_json_path, 'w', encoding='utf-8') as f:
                self.marked_key_dict = dict([(k, ','.join([str(i) for i in list(v)])) for k, v in vocab_list])
                f.write(json.dumps({'MARKED_KEYS': self.marked_key_dict}, indent=2))

        target_list = self.model.wv.index2word

        w2v_vocab_list = [(word, self.model.wv[word]) for word in self.model.wv.index2word if
                          word not in self.marked_key_dict]
        vocab_list = vocab_list + w2v_vocab_list

        if count:
            vocab_list = vocab_list[:count]

        self.embedding_matrix = np.zeros((len(vocab_list), self.embedding_size))

        for i in range(len(vocab_list)):
            word = vocab_list[i][0]
            self.idx2word[i] = word
            self.embedding_matrix[i] = vocab_list[i][1]

        self.word2idx = dict([(v, k) for (k, v) in self.idx2word.items()])

        marked_key_idx = list(self.idx2word.items())[:len(self.marked_key_dict)]
        logging.info('------------------------------')

        logging.info('Loaded gensim word2vec model')
        logging.info('model        : {}'.format(model_path))
        logging.info('word count   : {}'.format(len(w2v_vocab_list)))
        logging.info('special keys : {}'.format(marked_key_idx))
        logging.info('Top 50 word  : {}'.format(list(self.idx2word.items())[:50]))
        logging.info('------------------------------')

    def train(self,
              corpus_files,
              model_path,
              window=5,
              min_count=5,
              max_vocab_size=None,
              size=100,
              alpha=0.025,
              sample=1e-3,
              seed=1,
              min_alpha=0.0001,
              sg=0,
              hs=0,
              negative=5,
              cbow_mean=1,
              hashfxn=hash,
              iter=10,
              null_word=0,
              worker_mul=1):
        if not os.path.exists(model_path):
            os.mkdir(model_path)

        model_path = os.path.join(model_path, 'w2v.model')

        sentences = MySentences(corpus_files)

        logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
        logging.root.setLevel(logging.INFO)
        cores = multiprocessing.cpu_count() * worker_mul
        self.model = gensim.models.Word2Vec(sentences,
                                            size=self.embedding_size,
                                            window=window,
                                            min_count=min_count,
                                            max_vocab_size=max_vocab_size,
                                            alpha=alpha,
                                            sample=sample,
                                            seed=seed,
                                            min_alpha=min_alpha,
                                            sg=sg,
                                            hs=hs,
                                            negative=negative,
                                            cbow_mean=cbow_mean,
                                            hashfxn=hashfxn,
                                            iter=iter,
                                            null_word=null_word,
                                            workers=cores)
        self.model.save(model_path)

    def tokenize(self, text, with_mark=False):
        tokenized_text = []
        for word in text:
            tokenized_text.append(self.word2idx.get(word, self.word2idx[UNK]))
        if with_mark:
            tokenized_text.insert(self.word2idx[BOS], 0)
            tokenized_text.append(self.word2idx[EOS])
        return tokenized_text

    def reverse_tokenize(self, tokenized_text):
        text = []
        for token in tokenized_text:
            text.append(self.idx2word.get(token, UNK))
        return text


def parse_arg():
        """
        解析参数
        """
        parser = argparse.ArgumentParser()
        parser.add_argument('-i', '--input', dest='input', type=str)
        parser.add_argument('-o', '--output', dest='output', type=str, default='/home/bftv_lidan/')
        args = parser.parse_args()
        return vars(args)


def process():
    args = parse_arg()
    # args['input'] = '/Users/bftv_lidan/Desktop/segment_corpus/'
    # args['output'] = '/Users/bftv_lidan/Desktop/w2v'
    corpus = []
    for root, dirs, files in os.walk(args['input']):
        for file in files:
            filepath = os.path.join(root, file)
            corpus.append(filepath)
    if corpus:
        w = Word2Vec()
        print(corpus)
        w.train(corpus, model_path=args['output'])


if __name__ == "__main__":

    process()

    from nlp_toolkit.utils.tools import get_files_in_dir

    # corpus = ['/Users/leo/Desktop/ailab/chatbot-demo/single-round-chat/chinese/chat_1.yml']
    # corpus =[
    #     '/Users/bftv_lidan/Desktop/segment_corpus/segment_wechat_articles.txt',
    #     '/Users/bftv_lidan/Desktop/segment_corpus/segment_yingyin_movie.txt',
    #     '/Users/bftv_lidan/Desktop/segment_corpus/segment_douban_movie.txt',
    #     '/Users/bftv_lidan/Desktop/segment_corpus/segment_douban_people.txt',
    #     '/Users/bftv_lidan/Desktop/segment_corpus/segment_iqiyi_movie.txt',
    #     '/Users/bftv_lidan/Desktop/segment_corpus/segment_sogou_corpus.txt'
    # ]
    # corpus = []
    # for root, dirs, files in os.walk('/Users/bftv_lidan/Desktop/segment_corpus/'):
    #     for file in files:
    #         if file.startswith('.'):
    #             continue
    #         filepath = os.path.join(root, file)
    #         corpus.append(filepath)
    #         print(filepath)
    #
    # w = Word2Vec()
    # w.train(corpus, model_path='/Users/bftv_lidan/Desktop/w2v')
    # w.load_gensim('/Users/leo/Desktop/w2v2', count=100)
    # # print(w.tokenize('我 想 看 速度与激情'.split(' ')))
    #
    # # print(w.marked_key_dict)


