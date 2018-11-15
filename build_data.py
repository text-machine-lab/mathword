import os
import torch
from torch.utils.data import Dataset, DataLoader
import json
from random import shuffle
import nltk
import re
import itertools
import numpy as np
import argparse
from collections import OrderedDict
from word2number.w2n import word_to_num  # pip install word2number

import config
from src.word2vec import build_vocab
from transformer.Constants import *
from scripts.preprocess import reformat_equation

OPS = re.compile(r'([\+\-\*/\^\(\)=<>!;])', re.UNICODE)  # operators
DIGITS = re.compile(r'\d*\.?\d+')
WORDDIGITS = 'zero|one|two|three|four|five|six|seven|eight|nine'
WORDNUM = 'one|two|three|four|five|six|seven|eight|nine|ten| \
            eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen| \
            eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety'


# def cut_zeros(s):
#     s = re.sub(r'(\d*\.\d*[1-9])0+$', r'\1', s).rstrip('.')
#     if '.' in s:
#         return s.rstrip('0.')
#     return s


def equation_tokenize(equations, numbers):
    if not equations:
        return [], set([])
    equations = [reformat_equation(process_percent(s)) for s in equations]
    expr = ';'.join(equations)
    text_digits = [(v, k) for k, v in numbers.items()]
    text_digits.sort(key=lambda s: len(s), reverse=True)  # process long digits first, to avoid partial replace

    idx2key = {}
    unmached_digits = []
    unused = set(numbers.keys())
    while True:
        match = re.search(DIGITS, expr)
        if not match:
            break
        replaced = False
        start, end = match.span()
        for v, k in text_digits:
            try:
                v_equ, v_text = round(eval(match.group()), 3), round(eval(v), 3)
            except:  # not valid numbers
                continue
            if v_equ == v_text:
                if k in unused:
                    unused.remove(k)
                k_idx = 'N_' + chr(int(k[2:]) + ord('a'))  # so number index in N_1 won't be replaced as digits
                idx2key[k_idx] = k
                expr = expr[:start] + k_idx + expr[end:]
                replaced = True
                break  # replace one number only

        if not replaced:
            expr = expr[:start] + '#' + expr[end:]
            unmached_digits.append(match.group())
        else:
            # put it on the end of the list, so it has low priority to be used
            text_digits.remove((v, k))
            text_digits.append((v, k))

    for k_idx, k in idx2key.items():
        expr = expr.replace(k_idx, k)

    for item in unmached_digits:
        match = re.search(r'#', expr)
        start, end = match.span()
        expr = expr[:start] + item + expr[end:]

    expr = re.sub(OPS, r' \1 ', expr)

    return expr.split(), unused


def process_percent(s):
    matches = re.finditer(r'(\d+\.?\d*)(%|\spercent)', s.lower())
    offset = 0
    for match in matches:
        percent = float(match.group(1)) * 0.01
        start, end = match.span()
        insert = ' = {} '.format(percent)
        s = s[:end+offset] + insert + s[end+offset:]
        offset += len(insert)
    return s


def process_frac(s):
    matches = re.finditer(r'(\d+\s)?(\d+/\d+)', s)
    offset = 0
    for match in matches:
        start, end = match.span()
        integ, frac = match.group(1), match.group(2)
        if integ:
            num = int(integ)
        else:
            num = 0
        try:
            num += round(eval(frac), 3)
        except ZeroDivisionError:
            print(s)
            raise ZeroDivisionError
        except:
            return s
        insert = ' = {} '.format(num)
        s = s[:end+offset] + insert + s[end+offset:]
        offset += len(insert)
    else:
        return s


def word2digits(s):
    while True:
        match = re.search(r'(%s)+(\-(%s))?' %(WORDNUM, WORDDIGITS), s, re.IGNORECASE)
        if match:
            digits = word_to_num(match.group())
            start, end = match.span()
            s = s[:start] + str(digits) + s[end:]
        else:
            return s

def add_knowledge(s):
    text = s.lower()
    if 'triangle' in text and "angle" in text:
        s += ' The sum of angles is 180 degrees.'
    if 'circle' in text and "angle" in text:
        s += ' The sum of angles is 360 degrees.'
    if 'min' in text and 'hour' in text:
        s += ' 1 hour = 60 minutes.'
    if 'min' in text and 'sec' in text:
        s += ' 1 minute = 60 seconds.'
    if 'hour' in text and 'day' in text:
        s += ' 1 day = 24 hours.'
    if 'year' in text and 'month' in text:
        s += ' 1 year = 12 months.'
    if 'day' in text and 'month' in text:
        s += ' 1 month = 30 or 31 or 28 or 29 days.'
    if re.search(r'feet|foot|ft', text) or re.search(r'\d+\'\d+\"', text):
        s += ' 1 feet = 12 inches'
    if re.search(r'km|kilometer', text) and re.search(r'mile|mi\.', text):
        s += '1 mile = 1.608 km. 1 km = 0.621 miles'
    if re.search(r'lb|pound', text) and re.search(r'ounce|oz', text):
        s += '1 pound = 12 ounces'

def text_tokenize(question):
    def preprocess(question):
        question = word2digits(question)
        question = process_frac(question)
        question = process_percent(question)
        question = question.replace('/', ' / ')  # break fractions
        return question

    question = preprocess(question)
    words = nltk.word_tokenize(question)
    tokens = []
    numbers = OrderedDict()
    for word in words:
        if word[0] == '-' and len(word) > 1:
            tokens.append('-')
            word = word[1:]
        pattern0 = re.match(r'([a-zA-Z]+)(\d*\.?\d+)$', word)
        pattern1 = re.match(r'(\d*\.?\d+)\-?([a-zA-Z]+)$', word)
        if pattern0:
            tokens.append(pattern0.group(1))
            number = 'N_' + str(len(numbers))
            numbers[number] = pattern0.group(2)
            tokens.append(number)

        elif pattern1:
            number = 'N_' + str(len(numbers))
            numbers[number] = pattern1.group(1)
            tokens.append(number)
            tokens.append(pattern1.group(2))
        elif re.search(DIGITS, word.replace(',', '')):
            prev_end = 0
            for item in re.finditer(DIGITS, word.replace(',', '')):
                match = item.group()
                if match[0] == '0' and len(match) > 1 and '.' not in match:  # patterns like '025' not considered digits
                    tokens.append(match)
                    prev_end += len(match)
                    continue
                start, end = item.span()
                tokens += list(word[prev_end:start])
                number = 'N_' + str(len(numbers))
                numbers[number] = item.group()
                tokens.append(number)
                prev_end = end
            tokens += list(word[prev_end:])
        else:
            tokens.append(word)
    return tokens, numbers


def get_embedding_matrix(word_vectors=None):
    """
    special tokens:
    PAD_WORD = '<blank>'
    UNK_WORD = '<unk>'
    BOS_WORD = '<s>'
    EOS_WORD = '</s>'
    """
    n_special_toks = 4
    n_number_symbols = 10
    word_indexes = {}
    embedding_matrix = np.random.uniform(low=-0.5, high=0.5,
                                         size=(len(word_vectors) + n_special_toks + n_number_symbols,
                                               config.EMBEDDING_DIM))

    for index, word in enumerate(sorted(word_vectors.keys())):
        word_indexes[word] = index + n_special_toks
        embedding_vector = word_vectors.get(word, None)
        embedding_matrix[index + 4] = embedding_vector

    word_indexes[PAD_WORD] = 0
    word_indexes[UNK_WORD] = 1
    word_indexes[BOS_WORD] = 2
    word_indexes[EOS_WORD] = 3

    for i in range(n_number_symbols):
        word_indexes['N_'+str(i)] = len(word_indexes)

    embedding_matrix[0] = embedding_matrix[1] = embedding_matrix[2] = embedding_matrix[3] = np.zeros(config.EMBEDDING_DIM)
    embedding_matrix[1][1] = 1  # use one-hot for these special characters
    embedding_matrix[2][2] = 1
    embedding_matrix[3][3] = 1

    return embedding_matrix, word_indexes


def get_token_indexes(tokens):
    indexes = {}
    indexes[PAD_WORD] = 0
    indexes[UNK_WORD] = 1
    indexes[BOS_WORD] = 2
    indexes[EOS_WORD] = 3
    index = 4
    for tok in sorted(tokens):
        if tok not in indexes:
            indexes[tok] = index
            index += 1

    return indexes


def load_data(data_files, pretrained=True, max_len=200):
    data = []
    for f in data_files:
        data += json.load(open(f))
    shuffle(data)

    src = []
    tgt = []
    numbers = []
    index = {}
    src_truncated = 0
    tgt_truncated = 0
    for i, d in enumerate(data):
        text_toks, number_dict = text_tokenize(d['text'])
        if len(text_toks) > max_len:
            # print(text_toks)
            src_truncated += 1
            text_toks = text_toks[:max_len]

        try:
            equation_toks, unused = equation_tokenize(d['equations'], number_dict)
        except Exception:
            print(d)
            raise Exception
        if len(equation_toks) > max_len:
            # print(equation_toks)
            tgt_truncated += 1
            equation_toks = equation_toks[:max_len]

        for unused_number in unused:
            for x, tok in enumerate(text_toks):
                if tok == unused_number:
                    text_toks[x] = number_dict[tok]

        src.append(text_toks)
        tgt.append(equation_toks)
        numbers.append(number_dict)
        index[i] = d['id']

    print("src truncated {}, tgt truncated {}".format(src_truncated, tgt_truncated))
    if pretrained:
        src_vocab = build_vocab(itertools.chain(*src), config.WORD_VECTORS, K=50000)
        src_embeddings, src_indexes = get_embedding_matrix(word_vectors=src_vocab)
    else:
        src_indexes = get_token_indexes(itertools.chain(*src))

    tgt_indexes = get_token_indexes(itertools.chain(*tgt))

    data = {}
    data['dict'] = {}
    data['dict']['src'] = src_indexes
    data['dict']['tgt'] = tgt_indexes
    data['idx2id'] = index

    src_idx_repr = []
    for src_sent in src:
        idx = [BOS]
        idx += [src_indexes[s] if s in src_indexes else src_indexes[UNK_WORD] for s in src_sent]
        idx.append(EOS)
        src_idx_repr.append(idx)

    tgt_idx_repr = []
    for tgt_sent in tgt:
        idx = [BOS]
        idx += [tgt_indexes[s] for s in tgt_sent]
        idx.append(EOS)
        tgt_idx_repr.append(idx)

    data['src'] = src_idx_repr
    data['tgt'] = tgt_idx_repr
    data['numbers'] = numbers
    data['src_embeddings'] = src_embeddings
    data['settings'] = {}
    data['settings']['n_instances'] = len(src_idx_repr)
    data['settings']['src_vocab_size'] = len(src_indexes)
    data['settings']['tgt_vocab_size'] = len(tgt_indexes)
    data['settings']['max_src_seq'] = max([len(x) for x in src_idx_repr])
    data['settings']['max_tgt_seq'] = max([len(x) for x in tgt_idx_repr])
    data['settings']['max_token_seq_len'] = max([data['settings']['max_src_seq'], data['settings']['max_tgt_seq']])

    print(data['settings'])
    return data


def check_data(idx, data):
    src = data['src'][idx]
    tgt = data['tgt'][idx]
    if 'inv_dict' not in data:
        data['inv_dict'] = {}
        data['inv_dict']['src'] = {idx:word for word, idx in data['dict']['src'].items()}
        data['inv_dict']['tgt'] = {idx: word for word, idx in data['dict']['tgt'].items()}
    src_text = ' '.join([data['inv_dict']['src'][x] for x in src])
    tgt_text = ''.join([data['inv_dict']['tgt'][x] for x in tgt])
    return src_text, tgt_text


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-pretrained', action='store_true', default=False)
    parser.add_argument('--max-len', type=int, default=80)
    parser.add_argument('--dest', default='models/')
    args = parser.parse_args()

    pretrained = not args.no_pretrained
    # data = load_data(['/data2/ymeng/dolphin18k/formatted/eval_linear_auto_t1.json',
    #            '/data2/ymeng/dolphin18k/formatted/eval_linear_manual_t1.json'],
    #                  pretrained=pretrained, max_len=args.max_len)
    # data = load_data(['/data2/ymeng/dolphin18k/formatted/eval_linear_auto_t6.json',
    #                   '/data2/ymeng/dolphin18k/formatted/eval_linear_manual_t6.json'],
    #                  pretrained=pretrained, max_len=args.max_len)

    data = load_data(['/data2/ymeng/dolphin18k/eval_dataset/eval_dataset_formatted.json'],
                     pretrained=pretrained, max_len=args.max_len)
    path = os.path.join(args.dest, 'data.pt')
    torch.save(data, path)





