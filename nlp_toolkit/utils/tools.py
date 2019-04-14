# encoding: utf-8
"""
@author: leo 
@version: 1.0
@license: Apache Licence
@file: tools.py
@time: 2017/11/19 下午5:01

"""
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os


def filter_args(args):
    new = {}
    for k, v in args.items():
        if v != None:
            new[k] = v
    return new


def get_files_in_dir(directory, filetype=None):
    """
    This function will generate the file names in a directory
    tree by walking the tree either top-down or bottom-up. For each
    directory in the tree rooted at directory top (including top itself),
    it yields a 3-tuple (dirpath, dirnames, filenames).
    """
    file_paths = []

    # Walk the tree.
    for root, directories, files in os.walk(directory):
        for filename in files:
            if not filename.startswith('.'):
                if filetype:
                    if filename.endswith('.'+filetype):
                        filepath = os.path.join(root, filename)
                        file_paths.append(filepath)  # Add it to the list.
                else:
                    filepath = os.path.join(root, filename)
                    file_paths.append(filepath)  # Add it to the list.

    return file_paths


if __name__ == "__main__":
    print("Hello world")
