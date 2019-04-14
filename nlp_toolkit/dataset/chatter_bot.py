# encoding: utf-8
"""
@author: leo 
@version: 1.0
@license: Apache Licence
@file: chatter_bot
@time: 2018/1/26

"""


class ChatterBotData(object):
    def __init__(self, data_files, count=0):
        self.x_data = []
        self.y_data = []

        finished = False
        for file in data_files:
            for line in open(file, encoding='utf-8').read().splitlines()[3:]:
                new_line = line.replace('-', '').strip()
                new_data = [word for word in new_line.split(' ') if word != '']
                if line.startswith('  -'):
                    self.y_data.append(new_data)
                else:
                    self.x_data.append(new_data)

                if (count > 0) and (len(self.y_data) >= count):
                    finished = True
                    break
            if finished:
                break


if __name__ == "__main__":
    filepath = '/Users/leo/Desktop/ailab/chatbot-demo/single-round-chat/chinese/chat_1.yml'
    c = ChatterBotData(data_files=[filepath], count=10)
    for index in range(len(c.x_data)):
        print('x: {}'.format(c.x_data[index]))
        print('y: {}'.format(c.y_data[index]))

