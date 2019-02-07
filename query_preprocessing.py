
# coding: utf-8

# In[4]:


import random as rnd

import yargy
from yargy.tokenizer import MorphTokenizer
from yargy import Parser, rule, and_, or_, not_
from yargy.predicates import gram, dictionary, custom, true
from yargy.pipelines import morph_pipeline

import typing as t
import pymorphy2 as pmh
from pathlib import Path

import gensim.downloader as api
from pymystem3 import Mystem
import os
import pandas as pd
import numpy as np

import itertools as it

model = api.load("word2vec-ruscorpora-300")  # download the model and return as object ready for use

#проверка на то, число ли это
def is_number(string):
    for c in string:
        if((ord(c) < 48 or ord(c) > 57)):
            return False
    return True
	
is_number_ = custom(is_number)
#правило понимает дроби
NUMBER_RULE = rule(
    or_(
        gram("NUMR"),
        is_number_
    )
)
#все приставки, означающие денки:
MONEY_PIPE = morph_pipeline([
        "тыс",
        "к",
        "k",
        "м",
        "руб",
        "рублей",
        "тысяч"
])
#поиск токенов, означающих цену
#нижнюю границу
PRICE_FROM = rule(
    morph_pipeline([
        "от",
        "дороже"
    ]),
    NUMBER_RULE.repeatable(),
    MONEY_PIPE.optional().repeatable()
)
#верхнюю границу
PRICE_TO = rule(
    morph_pipeline([
        "до",
        "дешевле",
        "дешевле чем",
        "дешевле, чем"
    ]),
    NUMBER_RULE.repeatable(),
    MONEY_PIPE.optional().repeatable()
)
#точное значение
PRICE_VALUE = rule(
    NUMBER_RULE.repeatable(),
    not_(
        dictionary({
            "%",
            "процент",
            "процентов"
        })
    ),
    MONEY_PIPE.optional().repeatable()
)
#поиск атрибутов.
#Note: в строку атрибутов входит название самого товара
MEANING = rule(
    not_(
    or_(
        or_(
            or_(
                gram("INFN"),
                gram("VERB")
            ),
            or_(
                or_(
                    gram("PREP"), gram("CONJ")
                ),
                or_(
                    gram("PRCL"), gram("ADVB")
                )
            )
        ),
        gram('UNKN')
    )
    )
)
TRUE = rule(
    true
)
ATTRIBUTE = rule(
    MEANING
)
#поиск упоминаний процентов или денежных обозначений
MONEY_PERCENT = rule(
    or_(
    rule(
        morph_pipeline([
            "процент",
            "%"
        ]).optional(),
        MONEY_PIPE.repeatable()
        ),
    rule(
        morph_pipeline([
            "процент",
            "%"
        ]),
        MONEY_PIPE.optional().repeatable()
    )
    )
)
#упоминание о кэшбеке вместе с числовым значением
CASHBACK_PIPE = morph_pipeline([
        "кэшбек",
        "кэшбэк",
        "кешбек",
        "кешбэк",
        "кэшбека",
        "кэшбэка",
        "кешбека",
        "кешбэка",
        "cb",
        "кб",
        "кэш",
        "cashback",
        "кэшбеком",
        "кэшбэком",
        "кешбеком",
        "кешбэком"
])
#значение кэшбека
CASHBACK_VALUE = rule(
    NUMBER_RULE,
    MONEY_PERCENT.optional(),
)
CASHBACK_AFTER = rule(
    CASHBACK_PIPE,
    dictionary({
        "от",
        'с'
    }).optional(),
    NUMBER_RULE.optional().repeatable(),
    MONEY_PERCENT.optional()
)
CASHBACK_BEFORE = rule(
    dictionary({
        "от",
        'с'
    }).optional(),
    NUMBER_RULE.optional().repeatable(),
    MONEY_PERCENT.optional(),
    CASHBACK_PIPE
)
#число + обозначение процентов
PERCENT_RULE = rule(
    NUMBER_RULE,
    morph_pipeline([
        "%",
        "процент"
    ])
)
MONEY_RULE = rule(
    NUMBER_RULE.repeatable(),
    MONEY_PIPE.optional()
)

INSTALLMENT_PIPE = morph_pipeline([
    "в рассрочку",
    "рассрочка",
    "в кредит",
    "кредит"
])
IS_INSTALLMENT = rule(
    INSTALLMENT_PIPE
)

class Goods(object):
    def __init__(self, intent: str):
        self.analyzer = pmh.MorphAnalyzer()
        self.goods = []

        resource_directory = Path('./')
        self.paths = {
            'sport': resource_directory / 'sport.csv',
            'food': resource_directory / 'food.csv',
        }
        self.parse(self.paths[intent], ' ')
    def __getitem__(self, key):
        return self.goods[int(key)]
    #@overrides
    def parse(self, file: Path, bracket: str):
        #bracket - символ, отделяющий название от описания
        with file.open("r", encoding='utf-8') as file:
            parser = Parser(ATTRIBUTE)
            for line in file:
                line = line.replace('\n', '')
                self.goods.append(line)
                #print(line)
                for match in parser.findall(line):
                    for token in match.tokens:
                        self.goods.append(line[token.span.start:token.span.stop])
                        
        #исключаем повторы
        self.goods = list(set(self.goods))
        #print(self.goods)

money_value = {
    "k" : 1000,
    "к" : 1000,
    "тыс" : 1000,
    "тысяча" : 1000,
    "косарь" : 1000,#ХД
    "м" : 1000000,
    "миллион" : 1000000
}

class SlotFillerWithRules():
    def __init__(self):
        self.analyzer = pmh.MorphAnalyzer()
        self.price_rules = [PRICE_FROM, PRICE_TO]
        self.tokenizer = MorphTokenizer()
        self.dict = dict()
    def leveinstein_distance(self, str1, str2):
        "Calculates the Levenshtein distance between a and b."
        n, m = len(str1), len(str2)
        if n > m:
            str1, str2 = str2, str1
            n, m = m, n

        current_row = range(n+1) # Keep current and previous row, not entire matrix
        for i in range(1, m+1):
            previous_row, current_row = current_row, [i]+[0]*n
            for j in range(1,n+1):
                add, delete, change = previous_row[j]+1, current_row[j-1]+1, previous_row[j-1]
                if str1[j-1] != str2[i-1]:
                    change += 1
                current_row[j] = min(add, delete, change)

        return current_row[n]
    def preprocess(self, string):
        string = string.lower()
        string = ' '.join(self.analyzer.parse(token.value)[0].normal_form for token in self.tokenizer(string))
        string = " " + string + " "
        return string
    def parsing(self, string):
        parsed = dict()
        parsed['Offer_type'] = 0
        #FIND INSTALLMENT
        erased_string = string
        parser = Parser(IS_INSTALLMENT)
        for match in parser.findall(string):
            parsed['Offer_type'] = 1
            for token in match.tokens:
                erased_string = ' ' + erased_string.replace(" " + token.value + " ", " ") + ' '
        string = erased_string
        
        parsed['Cashback'] = "NaN"
        #find cashback with word 'cashback'
        cashback_rules = [CASHBACK_AFTER, CASHBACK_BEFORE]
        erased_string = string
        for rule in cashback_rules:
            if not (parsed['Cashback'] == "NaN" or parsed["Cashback"] == ""):
                break
            erased_string = string
            parser = Parser(rule)
            cashback_tokens = parser.findall(erased_string)
            cashback = ""
            #пока тренируемся на том, чnо кэшбек только на один товар
            for match in cashback_tokens:
                cashback += ' '.join([_.value for _ in match.tokens])
                if(cashback == ""):
                    continue
                for token in match.tokens:
                    erased_string = ' ' + erased_string.replace(" " + token.value + " ", " ") + ' '
            #вытаскиваем значения с размерностями:
            parser = Parser(CASHBACK_VALUE)
            cashback_tokens = parser.findall(cashback)
            cashback = ""
            for match in cashback_tokens:
                cashback += ' '.join([_.value for _ in match.tokens])
            #проверяем просто на вхождение процентов (т.к. пока мы рассрочку не учитываем)
            if(cashback == ""):
                parser = Parser(NUMBER_RULE)
                cashback_tokens = parser.findall(cashback)
                for match in cashback_tokens:
                    cashback += ' '.join([_.value for _ in match.tokens])
            else:
                parsed['Cashback'] = cashback.replace(" ", "")
                break
        string = erased_string.replace('[', '').replace(']', '')
        
        #FIND CASHBACK as %
        parser = Parser(PERCENT_RULE)
        percent_tokens = parser.findall(string)
        for match in percent_tokens:
            cashback = ' '.join([_.value for _ in match.tokens])
            #выбираем только числа без слов и знака %
            parser = Parser(NUMBER_RULE)
            for number_match in parser.findall(cashback):
                parsed['Cashback'] = ' '.join([_.value for _ in number_match.tokens])
            for token in match.tokens:
                string = string.replace(" " + token.value + " ", " ")
        
        #find
        parsed['Price_from'] = parsed['Price_to'] = 'NaN'
        price_keys = ['Price_from', 'Price_to']
        is_value = 0
        for i in range(2):
            parser = Parser(self.price_rules[i])
            price_tokens = parser.findall(string)
            for match in price_tokens:
                is_value += 1
                price_string = ' '.join([_.value for _ in match.tokens])
                parser = Parser(MONEY_RULE)
                money = ""
                for price_match in parser.findall(price_string):
                    money = ' '.join([_.value for _ in price_match.tokens])
                parsed[price_keys[i]] = money#' '.join([_.value for _ in match.tokens]).replace("до ", "").replace("до ", "")
                for token in match.tokens:
                    string = string.replace(" " + token.value + " ", " ")
        if (is_value == 0):
            parser = Parser(PRICE_VALUE)
            price_tokens = parser.findall(string)
            price = ""
            for match in price_tokens:
                price = ' '.join([_.value for _ in match.tokens])
                parsed['Price_from'] = parsed['Price_to'] = price
                for token in match.tokens:
                    string = string.replace(token.value + " ", "")
        #find ATTRIBUTE
        parser = Parser(ATTRIBUTE)
        attr = ""
        for match in parser.findall(string):
            attr += ' '.join([_.value for _ in match.tokens]) + ' '
        parsed['Attributes'] = attr[:-1]
        
        words = string.split(' ')
        parsed['Item'] = ""
        for word in words:
            #find Item
            #if(self.analyzer.parse(word)[0].normal_form in self.dict['goods']):
            #    parsed['Item'] += word + ' '
            #    #while True:
            #    #    pass
            
            #normalized_word = self.analyzer.parse(word)[0].normal_form
            normalized_word = word
            saved_word = ""
            minimum = len(normalized_word)
            maximum = 0
            max_word = ""
            is_noun = False
            for dictionary_word in self.dict['goods']:
                dis = self.leveinstein_distance(normalized_word, dictionary_word)
                if(dis < minimum and dis < min(len(dictionary_word), len(normalized_word)) / 2):
                    if(dis == 0):
                        max_word = dictionary_word
                        is_noun = False
                        for tags in self.analyzer.parse(dictionary_word):
                            if(tags.tag.POS == 'NOUN'):
                                is_noun = True
                                break
                        break
                    for tags in self.analyzer.parse(dictionary_word):
                        if(tags.score > maximum):
                            if(tags.tag.POS == 'NOUN'):
                                is_noun = True
                                max_word = dictionary_word
                            else:
                                is_noun = False
                                max_word = ""
                            
            if(is_noun):
                minimum = dis
                saved_word = max_word
        
            parsed['Item'] += saved_word + ' '
        words_a = parsed['Attributes'].split(' ')
        words_i = parsed['Item'].split(' ')
        for word in words_a:
            if(word in words_i):
                parsed['Attributes'] = parsed['Attributes'].replace(word, '')
        #parsed['Item'] = parsed['Item'][:-1]
        if(len(parsed['Item']) == 0):
            return parsed
        while parsed['Item'][0] == ' ':
            parsed['Item'] = parsed['Item'][1:]
            if(len(parsed['Item']) == 0):
                return parsed
        while parsed['Item'][-1] == ' ':
            parsed['Item'] = parsed['Item'][:-1]
            if(len(parsed['Item']) == 0):
                return parsed

        parsed['Item'] = parsed['Item'].strip()
        parsed['Attributes'] = parsed['Attributes'].strip()
         
        return parsed
    def fill(self, text: str, intent: str) -> t.Dict[str, t.Any]:
        self.dict['goods'] = Goods(intent)
        processed_string = self.preprocess(text)
        return self.normalize(self.parsing(processed_string))
    def normalize(self, form: t.Dict[str, t.Any]) -> t.Dict[str, t.Any]:
        keys = money_value.keys()
        price_keys = ['Price_from', 'Price_to']
        for key in price_keys:
            apokr = ""
            price = 0
            string = form[key]
            for sym in string:
                if(sym == " "):
                    continue
                if(ord(sym) >= 48 and ord(sym) <= 57):
                    price *= 10
                    price += int(sym)
                else:
                    apokr += sym
                    #основываемся на том, что все слова - значения порядка
                    if(apokr in keys):
                        price *= money_value[apokr]
                        apokr = ""
            form[key] = price
        if(form['Price_to'] == 0):
            form['Price_to'] = 999999999
        if(form['Cashback'] == '' or form['Cashback'] == 'NaN'):
            form['Cashback'] = 0
        else:
            cb_numbers = ""
            for sym in form['Cashback']:
                if(ord(sym) >= 48 and ord(sym) <= 57):
                    cb_numbers += sym
            form['Cashback'] = int(cb_numbers)
        return form
    
def isEnglish(s):
    if(s == ' '):
        return False
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True
    
def preprocess_text(str1):
    
    mystem = Mystem()
    tokens = mystem.lemmatize(str1.lower())
    str1 = " ".join(tokens)
    
    words = []
    for word in str1.split():
        if (word.isalpha()) and (not isEnglish(word)):
            words.append(word)
    
    res = set()
    for word in words:
        word_adv=word+'_ADJ'
        word_noun=word+'_NOUN'
        try:
            model.similarity(word_adv, 'слово_NOUN')
            res.add(word_adv)
        except BaseException:
            try:
                model.similarity(word_noun, 'слово_NOUN')
                res.add(word_noun)
            except BaseException:
                pass
    return res

def preproc(str1, _type):
    CUR_PATH = os.getcwd()
    try:
        os.chdir(CUR_PATH + '\\query_preprocessing')
    except BaseException:
        os.chdir(CUR_PATH + '//query_preprocessing')
    tmp = SlotFillerWithRules()
    res = tmp.fill(str1, _type)

    result = (list(preprocess_text(res['Attributes'] + ' ' + res['Item'])), res['Price_from'],res['Price_to'])
    
    os.chdir(CUR_PATH)
    return result