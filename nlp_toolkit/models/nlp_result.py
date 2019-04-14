#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : 2017-12-07 15:10:06
# @Author  : leo
# @Version : 1.0.0


class NLPResult(object):
    def __init__(self):
        self.intent = ''
        self.entities = []
        self.query = ''
        self.terms = []
        self.unknown_words = []
        self.spend_time = 0.0
        self.handle_method = ''
        self.rasa_entities = []

    def slu_format(self):
        return {
            'entities': self.entities,
            'rasa_entities': self.rasa_entities,
            'intent': self.intent,
            'query': self.query,
            'spend_time': self.spend_time,
            'terms': self.terms,
            'unknown_words': self.unknown_words,
            'handle_method': self.handle_method
        }
