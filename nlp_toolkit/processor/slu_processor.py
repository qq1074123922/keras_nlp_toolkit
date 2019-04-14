# encoding: utf-8
"""
@author: leo 
@version: 1.0
@license: Apache Licence
@file: slu_processor.py
@time: 2017/11/19 下午4:55

"""
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import json
import time
import sys
import argparse
import logging
import pickle
import numpy as np
import h5py
from tqdm import tqdm
from gensim.models import Word2Vec
from keras.preprocessing import sequence
import keras.utils.np_utils as np_utils

UNK = "#UNK"
PAD = "#PAD"
BOS = "#BOS"
EOS = "#EOS"


class MySentences(object):
    def __init__(self, files, tokenize_dict):
        self.files = files
        self.tokenize_dict = tokenize_dict

    def __iter__(self):
        """
        用于分行读取数据，训练词向量时候只需要一次读取一行，节省内存资源
        更多的请参照 yield 函数用法
        """
        for file in self.files:
            for line in open(file, encoding='utf-8'):
                # new_line = []
                # for word in line.split():
                #     if word in self.tokenize_dict:
                #         new_line.append(word)
                #     else:
                #         new_line.append(UNK)
                yield line.split()


class SLUProcessor(object):
    def __init__(self,
                 data_path,
                 corpus_path=None,
                 debug=False,
                 data_limit=1000000,
                 w2v_model=None):

        self.data_path = data_path
        self.input_path = os.path.join(self.data_path, 'input')
        self.output_path = os.path.join(self.data_path, 'output')
        self.model_path = os.path.join(self.data_path, 'model')
        self.dict_path = os.path.join(self.model_path, 'jieba')
        if os.path.exists(self.dict_path):
            self.jieba_dicts = [os.path.join(
                self.dict_path, f) for f in os.listdir(self.dict_path)]
        else:
            self.jieba_dicts = []

        self.add_on_corpus = []
        if corpus_path is not None:
            for f in os.listdir(corpus_path):
                if f.endswith('.txt'):
                    self.add_on_corpus.append(os.path.join(corpus_path, f))

        if debug:
            self.add_on_corpus = []

        self.words2idx = {}
        self.slots2idx = {}
        self.idx2slots = {}
        self.w2v_model = w2v_model

        self.train_x = os.path.join(self.input_path, 'train/words.txt')
        self.eval_x = os.path.join(self.input_path, 'eval/words.txt')
        self.test_x = os.path.join(self.input_path, 'test/words.txt')
        self.train_y = os.path.join(self.input_path, 'train/slots.txt')
        self.eval_y = os.path.join(self.input_path, 'eval/slots.txt')
        self.test_y = os.path.join(self.input_path, 'test/slots.txt')

        self.embedding_size = 100
        self.padding_size = 15
        self.embeddings_matrix = {}
        self.debug = debug
        self.data_limit = data_limit  # 一次 tokenize 数据数量
        self.random_idx = list()

    def path(self, sub_path):
        return os.path.join(self.model_path, sub_path)

    def read_base_info(self):
        """
        读取数据集基本信息
        """
        file = os.path.join(self.model_path, "dataset_info.pkl")
        with open(file, 'rb') as f:
            data = pickle.load(f)

        self.slots2idx = data["slots2idx"]
        self.words2idx = data["words2idx"]
        self.padding_size = data["padding_size"]
        self.embedding_size = data["embedding_size"]
        self.idx2slots = dict([(v, k) for k, v in self.slots2idx.items()])

    def read_model_conf(self, model_name='blstm'):
        file = os.path.join(self.model_path, 'trained_models',
                            model_name + '_model_config.json')
        with open(file, 'rb') as f:
            data = json.load(f)
        return data['model_conf'], data['train_info']

    def read_embeddings_matrix(self):
        """
        读取数据集 [id: word2vec] 词典，因为比较大单独处理，没有必要就不读取
        """
        h5f = h5py.File(os.path.join(
            self.model_path, "embeddings_matrix.h5"), 'r')
        self.embeddings_matrix = h5f["embeddings_matrix"][:]
        h5f.close()

    def train_word2vec(self):
        """
        训练词向量，并保存
        """
        model_path = os.path.join(self.output_path, 'w2v.model')
        corpus_files = [self.train_x, self.test_x] + self.add_on_corpus
        print('Start word2vec with files: ', corpus_files)
        sentences = MySentences(corpus_files, self.words2idx)

        logging.basicConfig(
            format='%(asctime)s: %(levelname)s: %(message)s')
        logging.root.setLevel(logging.INFO)
        logging.info("running %s" % ' '.join(sys.argv))
        model = Word2Vec(sentences, size=self.embedding_size,
                         window=5, min_count=5, workers=4)
        model.save(model_path)

        print("------------------------------------------------")
        print("Trained word2vec model    : ", model)
        print("------------------------------------------------")

        self.model = model
        self.words2idx = {
            PAD: 0
        }
        vocab_list = [(k, model.wv[k]) for k, v in model.wv.vocab.items()]
        embeddings_matrix = np.zeros(
            (len(model.wv.vocab.items()) + 1, self.embedding_size))
        for i in range(len(vocab_list)):
            word = vocab_list[i][0]
            self.words2idx[word] = i + 1
            embeddings_matrix[i + 1] = vocab_list[i][1]

        return embeddings_matrix

    def word2vec_to_embeddings_matrix(self):
        """
        训练词向量，并保存
        """
        model = self.w2v_model
        print("------------------------------------------------")
        print("Loaded word2vec model     : ", model)
        print("------------------------------------------------")

        self.model = model
        self.words2idx = {}

        special_keys = [PAD, UNK, BOS, EOS]

        vocab_list = []
        for index, key in enumerate(special_keys):
            self.words2idx[key] = index
            if index != 0:
                vector = np.random.uniform(-0.25, 0.25, self.embedding_size)
            else:
                vector = np.zeros(self.embedding_size)
            vocab_list.append((key, vector))

        w2v_vocab_list = [(k, model.wv[k]) for k, v in model.wv.vocab.items()]
        vocab_list = vocab_list + w2v_vocab_list
        embeddings_matrix = np.zeros((len(vocab_list), self.embedding_size))

        for i in range(len(vocab_list)):
            word = vocab_list[i][0]
            self.words2idx[word] = i
            embeddings_matrix[i] = vocab_list[i][1]

        return embeddings_matrix

    def save_embeddings_matrix(self, embeddings_matrix):
        meta_file = os.path.join(self.output_path, "w2v_metadata.tsv")
        word2idx = self.words2idx
        word2idx_sorted = [(k, word2idx[k]) for k in sorted(word2idx,
                                                            key=word2idx.get,
                                                            reverse=False)]
        print('Creating word2vec model metafile with vector: ', len(word2idx) - 1)
        with open(meta_file, 'wb') as file_metadata:
            for word in word2idx_sorted:
                # temporary solution for https://github.com/tensorflow/tensorflow/issues/9094
                if len(word[0]) == 0:
                    print("Emply Line, should replecaed by any thing else, "
                          "or will cause a bug of tensorboard")
                    file_metadata.write("{0}".format(
                        '<Empty Line>').encode('utf-8') + b'\n')
                else:
                    file_metadata.write("{0}".format(
                        word[0]).encode('utf-8') + b'\n')

        h5f = h5py.File(os.path.join(
            self.model_path, "embeddings_matrix.h5"), 'w')
        h5f.create_dataset("embeddings_matrix", data=embeddings_matrix)
        h5f.close()

    def tokenize_data(self):
        """
        数据的 token 处理，就是句子到 id 数组的完整过程
        """
        print("---------------------------------")
        print("Start tokenize data")
        padding = self.padding_size
        dataset_path = os.path.join(self.output_path, "dataset.h5")

        if os.path.exists(dataset_path):
            os.remove(dataset_path)
            print('Dataset file existed, deleted')

        def tokenize_file(file):
            new_file = []
            sample_x = []
            sample_y = []
            lines = open(file, 'r', encoding='utf-8').read().splitlines()
            # print('Tokenize file ', file)
            for i in tqdm(range(len(lines))):
                new_line = self.line_to_idx(lines[i].split())
                new_file.append(np.array(new_line))
                if len(sample_x) == 0:
                    sample_x, sample_y = lines[i].split(), new_line
            new_file = sequence.pad_sequences(new_file,
                                              maxlen=padding,
                                              dtype='int32',
                                              padding='pre')
            return np.array(new_file), sample_x, sample_y

        train_x, sample_x, sample_y = tokenize_file(self.train_x)
        eval_x, _, _ = tokenize_file(self.eval_x)
        test_x, _, _ = tokenize_file(self.test_x)

        print(
            "Tokenize data finished. Sample: {0} -> {1}".format(sample_x, sample_y))
        h5f = h5py.File(os.path.join(self.output_path, "dataset.h5"), "a")
        h5f.create_dataset("train_x", data=train_x)
        h5f.create_dataset("eval_x", data=eval_x)
        h5f.create_dataset("test_x", data=test_x)
        h5f.close()

    def tokenize_slots(self):
        """
        输出数据的 token 处理，就是句子到 id 数组的完整过程，
        由于 one hot 编码导致数据中有大量的 0，数据文件会比较大
        为了避免内存溢出，分批次处理并存储
        """
        print("---------------------------------")
        print("Start tokenize slots")
        padding = self.padding_size

        def tokenize_file(file, dictionary, h5f, dataset):
            lines = open(file, 'r', encoding='utf-8').read().splitlines()
            # 一次处理数据的量，如果内存不足可以适当增减少，内存充足则可以增大来提高处理速度
            handle_in_one_time = 5000
            run_count = int(len(lines) / handle_in_one_time + 1)
            h5f.create_dataset(dataset,
                               shape=(len(lines), self.padding_size,
                                      len(self.slots2idx)),
                               maxshape=(len(lines), self.padding_size,
                                         len(self.slots2idx)),
                               dtype=np.int32)

            sample_x = []
            sample_y = []

            for run in tqdm(range(run_count)):
                new_file = []
                for i in tqdm(range(handle_in_one_time)):
                    new_line = []
                    index = run * handle_in_one_time + i
                    if index < len(lines):
                        new_line = self.slots_to_idx(lines[index].split())
                        new_file.append(new_line)
                        if len(sample_x) == 0:
                            sample_x, sample_y = lines[index].split(), new_line
                chunck = sequence.pad_sequences(new_file,
                                                maxlen=padding,
                                                dtype='int32',
                                                padding='pre')
                new_file_1hot = []
                for index, line in enumerate(chunck):
                    y = np_utils.to_categorical(np.asarray(line)[:, np.newaxis],
                                                len(dictionary))
                    # h5f[dataset][run * handle_in_one_time + index] = y
                    new_file_1hot.append(y)
                max = handle_in_one_time * (run + 1)
                if max > len(lines):
                    max = len(lines)
                h5f[dataset][handle_in_one_time *
                             run:max] = np.asanyarray(new_file_1hot)
                # print('Tokenized {0:.2} of the slots'.format(run / run_count))
            print('Tokenized slots finished. Sample {0} -> {1}'
                  .format(sample_x, sample_y))

        dictionary = {}
        for file in [self.train_y, self.test_y]:
            for line in open(file):
                for w in line.split():
                    if w in dictionary:
                        dictionary[w] += 1
                    else:
                        dictionary[w] = 1
        slots2id = {PAD: 0}
        c = 0
        for k, v in dictionary.items():
            c += 1
            slots2id[k] = c
        self.slots2idx = slots2id
        slots2id_sorted = [(k, slots2id[k]) for k in sorted(slots2id,
                                                            key=slots2id.get,
                                                            reverse=False)]

        print("Slot tokenized finished, result: {0}".format(slots2id_sorted))

        h5f = h5py.File(os.path.join(self.output_path, "dataset.h5"), "a")
        tokenize_file(self.train_y, self.slots2idx, h5f, "train_y")
        tokenize_file(self.eval_y, self.slots2idx, h5f, "eval_y")
        tokenize_file(self.test_y, self.slots2idx, h5f, "test_y")
        h5f.close()

    def line_to_idx(self, sentence):
        """
        把具体某一行输入句子转行为 id 组成的数组，
        如 ['不要', '美国', '的', '_movie.query'] -> [225, 134, 6, 49]
        :param sentence: 句子
        :return: id 数组
        """
        new_line = []
        for w in sentence:
            if w in self.words2idx:
                new_line.append(self.words2idx[w])
            else:
                new_line.append(self.words2idx[UNK])
        return new_line

    def get_unkown_words(self, sentence):
        words = []
        for w in sentence:
            if w not in self.words2idx:
                words.append(w)
        return words

    def slots_to_idx(self, sentence):
        """
        把具体某一行输出 slots 转行为 id 组成的数组，
        如 ['O', 'O', 'O', 'area', 'movie_type'] -> [2, 2, 2, 4, 2, 9]
        :param sentence: 句子
        :return: id 数组
        """
        new_line = []
        for w in sentence:
            if w in self.slots2idx:
                new_line.append(self.slots2idx[w])
            else:
                Exception('不能存在这样的情况！！检查 slots2idx 字典')
        return new_line

    def idx_to_slots(self, idx):
        """
        把模型预测出来的 idx 转为一个可阅读数组，也就是 input_slots_to_idx 的反过程
        如 [2, 2, 2, 4, 2, 9] -> ['O', 'O', 'O', 'area', 'movie_type']
        :param idx: id 组成的数组
        :return: 可阅读的输出
        """
        new_line = []
        for w in idx:
            if w in self.idx2slots:
                new_line.append(self.idx2slots[w])
            else:
                Exception('不能存在这样的情况！！检查 slots2idx 字典')
        return new_line

    def save_info(self):
        data_set = {
            "slots2idx": self.slots2idx,
            "words2idx": self.words2idx,
            "padding_size": self.padding_size,
            "embedding_size": self.embedding_size
        }
        output = os.path.join(self.model_path, "dataset_info.pkl")
        with open(output, 'wb') as f:
            pickle.dump(data_set, f)

    def process(self):
        if self.w2v_model is None:
            embeddings_matrix = self.train_word2vec()
        else:
            embeddings_matrix = self.word2vec_to_embeddings_matrix()
        self.save_embeddings_matrix(embeddings_matrix)
        self.tokenize_data()
        self.tokenize_slots()
        self.save_info()


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', dest='dataset',
                        type=str, default=time.strftime('%Y.%m.%d'), help='dataset')
    parser.add_argument('-c', '--corpus', type=str, default=None)

    args = parser.parse_args()
    return vars(args)


if __name__ == "__main__":
    from gensim.models import KeyedVectors
    m = KeyedVectors.load_word2vec_format(
        '/Users/leo/Desktop/gensim_2017.11.22/w2v.raw.model')
    p = SLUProcessor(data_path='/Users/leo/Desktop/ailab/ailab_backend/data/generated_data/2017.11.24_nlp',
                     w2v_model=m)
    p.process()
    print('请直接调用 nlp_prepare_data.py')
