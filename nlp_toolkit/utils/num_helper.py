from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import re
import os


class NumDict(object):
    def get_num_data(self):
        data = [
            self.year_data(),
            self.gen_movie_name(),
            self.gen_data('', '', data_range=range(1, 300)),
            self.gen_data('第', '集'),
            self.gen_data('第', '个'),
            self.gen_data('第', '季'),
            self.gen_data(sub='分', data_range=range(1, 10)),
            self.gen_data(sub='分钟', data_range=range(1, 100)),
            self.gen_data(sub='小时', data_range=range(1, 10))
        ]
        dic = {}
        array = []
        for d, a in data:
            dic.update(d)
            array += a

        return dic, array

    def year_data(self):
        mapped = {
            '1': '一', '2': '二', '3': '三', '4': '四', '5': '五', '6': '六', '7': '七', '8': '八', '9': '九', '0': '零'
        }
        dic = {}
        array = []
        for y in range(1800, 2020):
            y_new = ''
            for s in str(y):
               y_new += mapped[s]
            dic[y_new + '年'] = [str(y), '年']
            dic[y_new] = [str(y)]
            array.append(str(y))
            array.append(y_new + '年')
            array.append(y_new)
        return dic, array

    def gen_data(self, pre = '', sub = '', data_range = range(1, 300)):
        dic = {}
        array = []
        for i in data_range:
            c = pre + NumberHelper.tocc(str(i)) + sub
            a = []
            for w in [pre, str(i), sub]:
                if w != '':
                    a.append(w)
            dic[c] = a
            array.append(str(i))
            array.append(c)
        return dic, array

    def gen_movie_name(self):
        dic = {}
        array = []
        # path = os.path.dirname(os.path.abspath(__file__)) + '/' + '../data/corpus/jieba_dicts/base_movie_name.csv'
        path = '/Users/bftv_lidan/Desktop/ailab/aida-api/data/jieba_dict/2018.03.17/dict_movie_name.txt'
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    if re.search(r'[0-9]', line.strip()):
                        try:
                            key1 = NumberHelper.tocc(line.strip(), full=True)  # 2015 -> 二千零一十五
                            key2 = NumberHelper.tocc_simple(line.strip())      # 2015 -> 二零一五
                        except ValueError:
                            pass

                        dic[key1] = [line.strip()]
                        dic[key2] = [line.strip()]
                        array.append(key1)
                        array.append(key2)

        return dic, array


class NumberHelper(object):
    chs_arabic_map = {u'零': 0, u'一': 1, u'二': 2, u'三': 3, u'四': 4,
                      u'五': 5, u'六': 6, u'七': 7, u'八': 8, u'九': 9,
                      u'十': 10, u'百': 100, u'千': 10 ** 3, u'万': 10 ** 4,
                      u'〇': 0, u'壹': 1, u'贰': 2, u'叁': 3, u'肆': 4,
                      u'伍': 5, u'陆': 6, u'柒': 7, u'捌': 8, u'玖': 9,
                      u'拾': 10, u'佰': 100, u'仟': 10 ** 3, u'萬': 10 ** 4,
                      u'亿': 10 ** 8, u'億': 10 ** 8, u'幺': 1,
                      u'０': 0, u'１': 1, u'２': 2, u'３': 3, u'４': 4,
                      u'５': 5, u'６': 6, u'７': 7, u'８': 8, u'９': 9}

    @staticmethod
    def tocc(words, full=False):
        '''数字转中文'''

        mapped = {
            '1': '一', '2': '二', '3': '三', '4': '四', '5': '五', '6': '六', '7': '七', '8': '八', '9': '九', '10': '十',
            '0': '零'
        }

        # 记录数字下标起始位
        number_index = ['NULL', 'NULL']
        # TODO 25届金鸡百花电影节暨第33 对此例子此处有bug，后面对该例跳过，暂不处理
        for k, v in enumerate(words):
            if mapped.get(v):
                if number_index[0] == 'NULL':
                    number_index[0] = k
                else:
                    number_index[1] = k

                    # 裸阿拉伯数字
        if number_index[0] != 'NULL' and number_index[1] != 'NULL':
            raw_data = words[number_index[0]:number_index[1] + 1]
        elif number_index[0] != 'NULL':
            raw_data = words[number_index[0]]
        else:
            return words

        # 应付上述bug, 对该例跳过不处理
        # try:
        #     int(raw_data)
        # except:
        #     return words

        r = ''

        # 处理 0 ~ 10
        if raw_data in mapped:
            r = mapped.get(raw_data)
        else:
            for k, v in [(1000, '千'), (100, '百'), (10, '十'), (0, '个')]:

                if k == 0:
                    # if len(raw_data) == 2 and r.startswith('一') and r.endswith('十'):
                    #     r = '十'
                    #     r += mapped.get(raw_data[-1])
                    #
                    # elif not r.endswith('十') and raw_data[-1] != '0':
                    #     r += '零' + mapped.get(raw_data[-1])
                    #
                    # elif raw_data[-1] == '0':
                    #     r = r
                    #
                    # else:
                    #     r += mapped.get(raw_data[-1])

                    #增加处理两种情况
                    if len(raw_data) == 2 and r.startswith('一') and r.endswith('十'):
                        r = '十'
                        r += mapped.get(raw_data[-1])
                    elif '千' in r and '十' in r and not '百' in r and raw_data[-1] != '0':
                        sep = r.index('千') + 1
                        pre = r[:sep]
                        post = r[sep:]
                        r = pre + '零' + post + mapped.get(raw_data[-1])
                    elif '千' in r and '十' in r and not '百' in r and raw_data[-1] == '0':
                        sep = r.index('千') + 1
                        pre = r[:sep]
                        post = r[sep:]
                        r = pre + '零' + post
                    elif raw_data[-1] != '0' and not r.endswith('十'):
                        r += '零' + mapped.get(raw_data[-1])
                    elif raw_data[-1] == '0':
                        r = r
                    else:
                        r += mapped.get(raw_data[-1])
                else:
                    # 处理千、百、十位
                    l = int(int(raw_data) % (k * 10) / k)
                    if l > 0:
                        string = mapped.get(str(l)) + v
                        r += string

        if full:
            tmp = list(words)
            if number_index[0] != 'NULL' and number_index[1] != 'NULL':
                tmp[number_index[0]:number_index[1] + 1] = r
            elif number_index[0] != 'NULL':
                tmp[number_index[0]] = r

            return str(''.join(tmp))

        return r

    @staticmethod
    def to_number(words, full=False):
        '''中文转数字'''

        mapped = {
            '一': '1', '二': '2', '三': '3', '四': '4', '五': '5', '六': '6', '七': '7', '八': '8', '九': '9',
            '零': '0', '十': '00', '百': '000', '千': '0000'
        }

        # 记录数字下标起始位
        number_index = ['NULL', 'NULL']
        for k, v in enumerate(words):
            if mapped.get(v):
                if number_index[0] == 'NULL':
                    number_index[0] = k
                else:
                    number_index[1] = k

                    # 裸中文数字
        if number_index[0] != 'NULL' and number_index[1] != 'NULL':
            raw_data = words[number_index[0]:number_index[1] + 1]
        elif number_index[0] != 'NULL':
            raw_data = words[number_index[0]]
        else:
            return words

        r = ''

        # 10 ~ 19 特殊处理
        if len(raw_data) == 1 and '十' in raw_data: r = '10'

        if len(raw_data) == 2 and raw_data[0] == '十': r = '1i'

        for item in raw_data:
            number = mapped.get(item)
            # 处理第一位
            print(words)
            print(number)
            if len(number) == 1 and not r:
                r += number

            # 处理千、百、十
            elif len(number) > len(r):
                rl = list(len(number) * 'i')
                for k, v in enumerate(r):
                    rl[k] = v

                    if raw_data[-1] == '十': rl[-1] = '0'
                    if raw_data[-1] == '百': rl[-2:] = '00'
                    if raw_data[-1] == '千': rl[-3:] = '000'

                    r = ''.join(rl)

            elif len(number) == 1:
                for k, v in enumerate(r):
                    try:
                        int(v)
                    except ValueError:
                        rl = list(r)
                        rl[k] = number
                        r = ''.join(rl)
                        break

        if full:
            tmp = list(words)
            if number_index[0] != 'NULL' and number_index[1] != 'NULL':
                tmp[number_index[0]:number_index[1] + 1] = r
            elif number_index[0] != 'NULL':
                tmp[number_index[0]] = r

            return str(''.join(tmp))

        return r

    @staticmethod
    def tocc_simple(words):
        '''数字转中文, 一一对应'''

        mapped = {
            '1': '一', '2': '二', '3': '三', '4': '四', '5': '五', '6': '六', '7': '七', '8': '八', '9': '九', '0': '零'
        }

        # 提取阿拉伯数字
        result = ''.join([mapped.get(item, item) for item in words])

        return result

    @staticmethod
    def c2a(chinese_digits):
        result = 0
        tmp = 0
        hnd_mln = 0
        for count in range(len(chinese_digits)):
            curr_char = chinese_digits[count]
            curr_digit = NumberHelper.chs_arabic_map.get(curr_char, None)
            # meet 「亿」 or 「億」
            if curr_digit == 10 ** 8:
                result = result + tmp
                result = result * curr_digit
                # get result before 「亿」 and store it into hnd_mln
                # reset `result`
                hnd_mln = hnd_mln * 10 ** 8 + result
                result = 0
                tmp = 0
            # meet 「万」 or 「萬」
            elif curr_digit == 10 ** 4:
                result = result + tmp
                result = result * curr_digit
                tmp = 0
            # meet 「十」, 「百」, 「千」 or their traditional version
            elif curr_digit >= 10:
                tmp = 1 if tmp == 0 else tmp
                result = result + curr_digit * tmp
                tmp = 0
            # meet single digit
            elif curr_digit is not None:
                tmp = tmp * 10 + curr_digit
            else:
                return result
        result = result + tmp
        result = result + hnd_mln
        return result

    @staticmethod
    def convert_sentence(text, reverse=False, debug=False):
        """自动中汉数字转换, 提取"""

        prefix = ['评分', '评价', '第']
        subfix = ['年', '月', '年代', '个', '分钟', '分']
        numbers = '零一二两三四五六七八九十百千万亿'

        def convery(text):
            result = list(re.finditer('[零一二两三四五六七八九十百千万亿]+', text))
            for i in result:
                if text[i.start()] in prefix or text[i.end()] in subfix:
                    a_num = NumberHelper.c2a(i.group())
                    text = NumberHelper.re_replacer(text, i.span(), str(a_num))
            return text

        convery(text)

        try:
            convery(text)
        except:
            print('转换数字失败，原文：{0}'.format(text))

    @staticmethod
    def convert_word(text):
        """自动中汉数字转换, 提取"""
        numbers = '零一二两三四五六七八九十百千万亿'
        result = list(re.finditer('[{0}]+'.format(numbers), text))
        for i in result:
            a_num = NumberHelper.c2a(i.group())
            text = NumberHelper.re_replacer(text, i.span(), str(a_num))
        return text

    @staticmethod
    def re_replacer(text, sub, new):
        start, end = sub
        return text[:start] + new + text[end:]


if __name__ == '__main__':
    # print('\n阿拉伯数字转中文: \n')
    helper = NumberHelper()
    # print('0\t\t-->\t', toCc('0', full=True))
    # print('第11集\t\t-->\t', helper.tocc('25届金鸡百花电影节暨第33', full=True))
    # print('第100集\t\t-->\t', toCc('第100集', full=True))
    # print('第101集\t\t-->\t', toCc('第101集', full=True))
    # print('101次求婚\t-->\t', toCc('101次求婚', full=True))
    # print('101次求婚\t-->\t', toCc('战狼2', full=True))

    # print('\n中文转阿拉伯数字: \n')
    # print('零\t\t-->\t', toNumber('零', full=True))
    # print('第十一集\t-->\t', toNumber('第十一集', full=True))
    # print('第一百集\t-->\t', toNumber('第一百集', full=True))
    # print('第一百零一集\t-->\t', toNumber('第一百零一集', full=True))
    # print('一百零一次求婚\t-->\t', toNumber('一百零一次求婚', full=True))

    # print(NumberHelper.to_number('速度与激情二', full=True))
    # print(NumberHelper.to_number('25届金鸡百花电影节暨第33'))
    # print(convert_sentence('第一百零一集'))
    # print(convert_sentence('二'))
    # print(convert_sentence('想看小猪佩奇四'))

    # print(os.path.abspath(__file__))
    # path = os.path.dirname(os.path.abspath(__file__)) + '/' + '../data/corpus/jieba_dicts/base_movie_name.csv'
    # print(path)
    # print(os.path.exists(path))

    # from pypinyin import pinyin, lazy_pinyin, Style
    #
    # word = '你好'
    # print(pinyin(word, style=Style.TONE2, heteronym=False))
    # print(lazy_pinyin(word))