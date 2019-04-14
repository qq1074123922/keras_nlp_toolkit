# encoding: utf-8
"""
@author: leo 
@version: 1.0
@license: Apache Licence
@file: slu_trainer.py
@time: 2017/11/19 下午4:59

"""
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import json
import time
import os
import h5py
import random
from shutil import copyfile

import numpy as np
from bf_nlp.models.slu_model import JointSLUModel
from bf_nlp.processor.slu_processor import SLUProcessor
from keras.callbacks import TensorBoard, ModelCheckpoint, RemoteMonitor, Callback, LambdaCallback


class SLUTrainer(object):
    def __init__(self, data_path, **conf):
        self.prosessor = SLUProcessor(data_path=data_path)
        self.model = JointSLUModel(data_path=data_path, **conf)
        self.model.build()
        self.train_info = {
            'score': 0,
            'train_time': 0,
            'trained_at': time.time()
        }

    def save_info(self):
        data = {
            'model_conf': self.model.conf,
            'train_info': self.train_info
        }
        model_config_json = self.model.model_base_path + '_model_config.json'
        with open(model_config_json, 'w') as outfile:
            outfile.write(json.dumps(data, indent=4, ensure_ascii=False))

    def train(self):
        h5_path = os.path.join(self.prosessor.output_path, 'dataset.h5')
        h5f = h5py.File(h5_path, 'r')
        start_train = time.time()

        print('\n+++++++++++++++++++++++++++++++++++++++++++++++')
        print('Start train {0} model with dataset {1}'.format(
            self.model.out, self.prosessor.model_path))
        print('Nums of slot: ', len(self.prosessor.slots2idx))
        print('Nums of sentence use for training: ',
              format(h5f['train_x'].len(), ','))
        print('Train batch size: ', self.model.batch_size)
        print('Run `tensorboard --logdir={0}/logs` to visualize on tensorboard'.format(
            self.prosessor.model_path))
        print('\n+++++++++++++++++++++++++++++++++++++++++++++++')

        # saves the model weights after each epoch if the validation loss decreased
        check_pointer = ModelCheckpoint(filepath=self.model.model_path,
                                        verbose=1,
                                        save_best_only=True)
        # saves the model weights after each epoch if the validation loss decreased
        log_dir = os.path.join(self.prosessor.model_path, "logs", self.model.out)
        try:
            os.remove(log_dir)
        except:
            pass
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        w2v_metadata = os.path.join(
            self.prosessor.output_path, 'w2v_metadata.tsv')
        copyfile(w2v_metadata, os.path.join(log_dir, 'w2v_metadata.tsv'))
        print(os.path.join(log_dir, 'w2v_metadata.tsv'))

        embeddings_freq = 1
        if len(self.model.embeddings_matrix) == 0:
            embeddings_freq = 3
        log_call_back = TensorBoard(log_dir=log_dir,
                                    histogram_freq=1,
                                    batch_size=self.model.batch_size,
                                    write_graph=False,
                                    write_images=False,
                                    embeddings_freq=embeddings_freq,
                                    embeddings_layer_names=None,
                                    embeddings_metadata="w2v_metadata.tsv")

        self.train_info['trained_sentence'] = h5f["train_x"].len()
        self.save_info()

        eval_x, eval_y = h5f["eval_x"][:], h5f["eval_y"][:]
        test_x, test_y = h5f["test_x"][:], h5f["test_y"][:]
        data_count = h5f["train_x"].len()
        # 根据数据数量，采取不同的读取方式，数据少直接全部加载到内存

        if self.model.use_generator:
            print('Start training with model.fit_generator')
            print('-----------------------------------------')

            # 从节省内存的角度，通过生成器的方式来训练

            def data_generator(data, targets, batch_size):
                data_count = len(data)
                loopcount = data_count // batch_size + 1
                while True:
                    i = random.randint(0, loopcount)
                    start_index = i * batch_size
                    end_index = (i + 1) * batch_size
                    if end_index > data_count:
                        end_index = data_count
                        start_index = data_count - batch_size
                    yield data[start_index:end_index], targets[start_index:end_index]

            generator = data_generator(
                h5f["train_x"], h5f["train_y"], self.model.batch_size)
            self.model.model.fit_generator(generator,
                                     steps_per_epoch=h5f["train_x"].len(
                                     ) // self.model.batch_size,
                                     epochs=self.model.epochs,
                                     callbacks=[log_call_back, check_pointer],
                                     validation_data=(eval_x, eval_y))
        else:
            print('Start training with model.fit')
            print('-----------------------------------------')
            train_x, train_y = h5f["train_x"][:], h5f["train_y"][:]
            self.model.fit(train_x, train_y,
                           batch_size=self.model.batch_size,
                           epochs=self.model.epochs,
                           verbose=1,
                           callbacks=[log_call_back, check_pointer],
                           validation_data=(eval_x, eval_y))
        print('-----------------------------------------')
        h5f.close()
        score = self.model.model.evaluate(test_x, test_y,
                                    batch_size=500,
                                    verbose=1)
        spend_time = time.time() - start_train
        self.train_info['score'] = score
        self.train_info['train_time'] = spend_time
        self.save_info()
        print('\n------------------------------------------------')
        print('Finished precess dataset'.format(self.prosessor.model_path))
        print('Nums of slot: ', len(self.prosessor.slots2idx))
        print('Nums of sentence use for training: ', format(data_count, ','))
        print('Total spend time: {0:.8} s'.format(spend_time))
        print('Final test score:', score)
        print('Run `tensorboard --logdir={0}/logs` to visualize on tensorboard'.format(
            self.prosessor.model_path))
        print('------------------------------------------------')
        return score, spend_time

    def get_random_data(self, data_list, amount):
        idx = np.random.choice(
            np.arange(len(data_list[0])), int(amount), replace=False)
        data_list_result = []
        for d in data_list:
            n = []
            for i in idx:
                n.append(d[i])
                # print(d[i])
            data_list_result.append(np.asanyarray(n))
        return data_list_result


if __name__ == "__main__":
    t = SLUTrainer('/Users/leo/Desktop/ailab/ailab_backend/data/generated_data/2017.11.13_nlp')
    t.train()
    print("Hello world")
