# encoding: utf-8
"""
@author: leo 
@version: 1.0
@license: Apache Licence
@file: encoder.py
@time: 2017/11/19 下午3:29

"""
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import json
import numpy as np


class NumpyEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)


if __name__ == "__main__":
    print("Hello world")
