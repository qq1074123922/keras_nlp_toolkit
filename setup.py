# encoding: utf-8
"""
@author: leo
@version: 1.0
@license: Apache Licence
@file: setup.py.py
@time: 2017/11/16 下午1:35

"""

from setuptools import setup

__version__ = None  # Avoids IDE errors, but actual version is read from version.py
exec(open('nlp_toolkit/version.py', encoding='utf-8').read())

install_requires = [
    "jieba",
    "numpy",
    "pymysql",
    "h5py",
    "keras",
    "tqdm",
    "gensim",
    "tensorflow"
]

setup(
    name='nlp_toolkit',
    packages=[
        'nlp_toolkit',
        'nlp_toolkit.agent',
        'nlp_toolkit.dataset',
        'nlp_toolkit.models',
        'nlp_toolkit.processor',
        'nlp_toolkit.resource',
        'nlp_toolkit.utils',
    ],
    classifiers=[
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6"
    ],
    version=__version__,
    install_requires=install_requires,
    include_package_data=True,
    description="Baofeng NLP Toolkit",
    author='leo'
)

print("\nWelcome to Rasa NLU!")
print("If any questions please visit documentation page https://rasahq.github.io/rasa_nlu")
print("or join community chat on https://gitter.im/RasaHQ/rasa_nlu")
