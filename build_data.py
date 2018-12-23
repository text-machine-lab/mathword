import os
import torch
import json
from random import shuffle
import nltk
import re
import itertools
import numpy as np
import argparse
from collections import OrderedDict, Counter
from word2number.w2n import word_to_num  # pip install word2number

import config
from src.word2vec import build_vocab, get_glove
from transformer.Constants import *
from scripts.preprocess import reformat_equation

OPS = re.compile(r'([\+\-\*/\^\(\)=<>!;])', re.UNICODE)  # operators
DIGITS = re.compile(r'\d*\.?\d+')
WORDDIGITS = 'zero|one|two|three|four|five|six|seven|eight|nine'
WORDNUM = 'one|two|three|four|five|six|seven|eight|nine|ten| \
            eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen| \
            eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety'
COMMON_CONSTANTS = set(['0.5', '1', '2', '12', '24', '10', '100', '1000', '60', '180', '360', '3.14', '3.1416'])
MATH_TOKENS = ['+', '-', '*', '/', '=', '(', ')', ';', '^', 'sqrt', 'sin', 'cos', 'tan', 'cot', 'exp']
UNITS = ['m', 'cm', 'mm', 'ft', 'inch', 'mph', 'g', 'kg', 'mg', 'lb', 'lbs', 'oz', 'mi', 'rad', '\u00b0']
N_SYMBOLS = 10


def equation_tokenize(expr, numbers):
    if not expr:
        return [], set([])
    text_digits = [(v, k) for k, v in numbers.items()]
    text_digits.sort(key=lambda s: len(s), reverse=True)  # process long digits first, to avoid partial replace

    idx2key = {}
    unmached_digits = []
    unused = set(numbers.keys())

    expr_copy = expr
    for match in re.finditer(DIGITS, expr_copy):
        try:
            expr = expr.replace(match.group(), str(round(eval(match.group()), 3)), 1)
        except SyntaxError:
            print(expr_copy, expr, match.group)
            assert SyntaxError

    while True:
        match = re.search(DIGITS, expr)
        if not match:
            break
        replaced = False
        start, end = match.span()
        for v, k in text_digits:
            try:
                v_equ, v_text = eval(match.group()), round(eval(v), 3)
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


def process_percent(s, equations):
    matches = re.finditer(r'(\d+\.?\d*)(\s?%|\spercent)', s.lower())
    offset = 0
    for match in matches:
        numerator = match.group(1)
        percent = round(float(numerator) * 0.01, 3)
        start, end = match.span()

        if re.search(r'(^|[^\d]){}0*([^\d]|$)'.format(percent), equations):
            insert = '{}'.format(percent)
        else:
            insert = '{}/100'.format(numerator)
        s = s[:start + offset] + insert + s[end + offset:]
        offset += len(insert) - end + start

    return s


def process_frac(s, equations):
    """ try keeping original forms from equations. If not found in equations:
        1/3 -> 1/3 (0.333)
        2 1/3 -> 7/3 (2.333)
    """
    matches = re.finditer(r'(\d+\s+)?(\d+/\d+)', s)
    offset = 0
    for match in matches:
        start, end = match.span()
        start += offset
        end += offset
        integ, frac = match.group(1), match.group(2)
        if integ:
            num = int(integ)
        else:
            num = 0
        try:
            value = num + round(eval(frac), 3)
        except ZeroDivisionError:
            print(s)
            raise ZeroDivisionError
        except:
            return s, equations

        numerator, denominator = frac.split('/')
        numerator = num * int(denominator) + int(numerator)

        # print(match.group())

        if end < len(s) -2 and s[end+1] == '%' and \
                re.search(r'(^|[^\d]){}([^\d]|$)'.format(round(value / 100, 3)), equations):
            insert = '{}'.format(round(value / 100, 3))
            end +=2

        elif re.search(r'(^|[^\d]){}/{}([^\d]|$)'.format(numerator, denominator), equations):
            # insert = '{}/{}'.format(numerator, denominator)
            insert = '{}'.format(value)
            equations = re.sub(r'(^|[^\d]){}/{}([^\d]|$)'.format(numerator, denominator), r'\g<1>{}\2'.format(value), equations)

        elif re.search(r'(^|[^\d]){}([^\d]|$)'.format(value), equations):
            insert = '{}'.format(value)
        elif end < len(s) -2 and s[end+1] == '%':
            insert = '{} %'.format(value)
            end += 2
        else:
            insert = '{}/{}'.format(numerator, denominator)
        # print(insert)
        s = s[:start] + insert + s[end:]
        offset += len(insert) - end + start

    return s, equations


def word2digits(s):

    s = re.sub(r'twice', '2 times', s, flags=re.IGNORECASE)
    s = re.sub(r' quarter', ' quarter (1/4 = 0.25)', s, flags=re.IGNORECASE)
    s = re.sub(r' half ', ' half (1/2 = 0.5) ', s, flags=re.IGNORECASE)
    s = re.sub(r'one[\s\-]*third', '1/3', s, flags=re.IGNORECASE)
    s = re.sub(r'tow[\s\-]*thirds', '2/3', s, flags=re.IGNORECASE)
    s = re.sub(r'one[\s\-]*fourth', '1/4', s, flags=re.IGNORECASE)
    s = re.sub(r'three[\s\-]*fourths', '3/4', s, flags=re.IGNORECASE)
    s = re.sub(r'one[\s\-]*fifth', '1/5', s, flags=re.IGNORECASE)

    s = re.sub(r'(\d+\.?\d*)\scents', r'$\1%', s, flags=re.IGNORECASE)

    while True:
        match = re.search(r'(\s|^)((%s)+(\-(%s))?)([^\w]|$)' %(WORDNUM, WORDDIGITS), s, re.IGNORECASE)
        # print( match.group(2), match.group(6))
        if match and match.group(2) not in ('one', 'One'):
            digits = word_to_num(match.group(2))
            start, end = match.span()
            if start != 0:
                start += 1
            s = s[:start] + str(digits) + s[end-1:]
        else:
            break
    match = re.search(r'(\d+)\shundred', s)
    if match:
        val = eval(match.group(1)) * 100
        s = s.replace(match.group(), str(val))
    match = re.search(r'(\d+)\sthousand', s)
    if match:
        val = eval(match.group(1)) * 1000
        s = s.replace(match.group(), str(val))

    return s

def add_knowledge(s):
    text = s.lower()
    if 'deck' in text and "cards" in text:
        s += ' 1 deck = 52 cards. 1 suit = 13 cards.'
    if 'triangle' in text and "angles" in text:
        s += ' The sum of angles is 180 degrees.'
    if re.search(r'square|rectangular|quadrilateral', text) and "angle" in text:
        s += ' The sum of angles is 360 degrees.'
    if 'circle' in text and "angle" in text:
        s += ' The sum of angles is 360 degrees.'
    if 'minute' in text:
        s += ' 1 hour = 60 minutes = 3600 seconds.'
    if 'min' in text and 'sec' in text:
        s += ' 1 minute = 60 seconds.'
    if 'hour' in text and 'day' in text:
        s += ' 1 day = 24 hours.'
    if re.search(r'year|annual', text) and 'month' in text:
        s += ' 1 year = 12 months.'
    if 'day' in text and 'month' in text:
        s += ' 1 month = 30 or 31 or 28 or 29 days.'
    if (re.search(r'feet|foot|ft', text) and re.search(r'inch|in\.', text)) or re.search(r'\d+\'\d+\"', text):
        s += ' 1 feet = 12 inches'
    if re.search(r'km|kilometer', text) and re.search(r'mile|mi\.', text):
        s += ' 1 mile = 1.608 km. 1 km = 0.621 miles'
    if re.search(r'meter', text) and re.search(r'ft|feet|foot', text):
        s += ' 1 m = 3.28 ft'
    if re.search(r'meter|\d\s?m', text) and re.search(r'\d\s?cm|\d\s?mm', text):
        s += ' 1 m = 100 cm = 1000 mm'
    if re.search(r'kg|kilogram', text) and re.search(r'[\s\d](gram|g\s[\.\?])', text):
        s += ' 1 kg = 1000 g'
    if re.search(r'[\s\d]mg\s[\.\?]', text) and re.search(r'[\s\d](gram|g\s[\.\?])', text):
        s += ' 1 g = 1000 mg'
    if re.search(r'liter|litre|[\s\d]l[\s\.]', text) and re.search(r'mililit|ml', text):
        s += ' 1 L = 1000 mL'
    if re.search(r'feet|ft', text) and re.search(r'mile|mi\.', text):
        s += ' 1 mile = 5280 feet'
    if re.search(r'lb|pound', text) and re.search(r'ounce|oz', text):
        s += ' 1 pound = 16 ounces'
    if re.search(r'yd|yard', text):
        s += ' 1 yard = 3 feet = 36 inches'
    if 'centi' in text:
        s += 'centi = 1/100'
    if 'milli' in text:
        s += 'milli = 1/1000'
    if 'rad' in text and 'degree' in text:
        s += '1 radian = 180/3.14 degrees'

    return s


def remove_choices(question):
    return re.sub(r'[\(\s]\s?A[\s\.\)].{1,15}[\(\s]\s?B[\s\.\)].{1,15}[\(\s]\s?C[\s\.\)].{1,15}[\(\s]\s?D[\s\.\)].+',
                  '', question, flags=re.IGNORECASE)

def text_tokenize(question, equations):
    # preprocess
    question = remove_choices(question)
    question = word2digits(question)
    question, equations = process_frac(question, equations)
    question = process_percent(question, equations)
    question = question.replace('/', ' / ')  # break fractions
    question = add_knowledge(question)

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
            word = word.replace(',', '')
            for item in re.finditer(DIGITS, word):
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
            if prev_end != len(word):
                # print('last', prev_end, len(word))
                tokens += list(word[prev_end:])
        else:
            tokens.append(word)
    return tokens, numbers, equations


def get_embedding_matrix(word_vectors=None):
    """
    special tokens:
    PAD_WORD = '<blank>'
    UNK_WORD = '<unk>'
    BOS_WORD = '<s>'
    EOS_WORD = '</s>'
    """
    n_special_toks = 4
    n_number_symbols = N_SYMBOLS
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
        # embedding_matrix[len(word_indexes)] = np.zeros(config.EMBEDDING_DIM)
        # embedding_matrix[len(word_indexes)][100] = 1   # set special values for N_x... they are all the same
        word_indexes['N_' + str(i)] = len(word_indexes)

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


def get_vocab(tokens, glove_path, cutoff=3):
    counter = Counter(tokens)
    word_dict = {}
    for tok, freq in counter.items():
        if freq >= cutoff:
            word_dict[tok] = ''
    math_tokens = [str(x) for x in range(10)] + list(COMMON_CONSTANTS) + MATH_TOKENS + UNITS
    for tok in math_tokens:
        word_dict[tok] = ''
    word_vec = get_glove(word_dict, glove_path)
    print('Vocab size : {0}'.format(len(word_vec)))
    return word_vec


def load_data(data_files, pretrained=True, max_len=200):
    data = []
    for f in data_files:
        data += json.load(open(f))
    shuffle(data)

    src = []
    tgt = []
    tgt_num = []
    answers = []
    numbers = []
    index = {}
    src_truncated = 0
    tgt_truncated = 0
    for i, d in enumerate(data):
        equations = ';'.join([reformat_equation(s) for s in d['equations']])
        text_toks, number_dict, equations = text_tokenize(d['text'], equations)
        if len(text_toks) > max_len:
            src_truncated += 1
            text_toks = text_toks[:max_len]

        equation_toks, unused = equation_tokenize(equations, number_dict)

        if len(equation_toks) > max_len:
            # print(equation_toks)
            tgt_truncated += 1
            equation_toks = equation_toks[:max_len]
        tgt_num_tokens= [x for x in equation_toks if x[0]=='N' or x in COMMON_CONSTANTS]  # constants or N-x tokens in equation

        for unused_number in unused:
            for x, tok in enumerate(text_toks):
                if tok == unused_number and tok in COMMON_CONSTANTS:
                    text_toks[x] = number_dict[tok]

        src.append(text_toks)
        tgt.append(equation_toks)
        tgt_num.append(tgt_num_tokens)
        answers.append(d['ans'])
        numbers.append(number_dict)
        index[i] = d['id']

    print("src truncated {}, tgt truncated {}".format(src_truncated, tgt_truncated))
    if pretrained:
        # src_vocab = build_vocab(itertools.chain(*src), config.WORD_VECTORS, K=50000)
        src_vocab = get_vocab(itertools.chain(*src), config.WORD_VECTORS, cutoff=8)
        special_symbols = ['N_' + str(i) for i in range(N_SYMBOLS)] + [PAD_WORD, UNK_WORD, BOS_WORD, EOS_WORD]
        for symbol in special_symbols:
            if symbol in src_vocab:
                del src_vocab[symbol]
                print("delete token {} before setting word indexes".format(symbol))
        src_embeddings, src_indexes = get_embedding_matrix(word_vectors=src_vocab)
    else:
        src_indexes = get_token_indexes(itertools.chain(*src))

    # tgt_indexes = get_token_indexes(itertools.chain(*tgt))
    tgt_tokens = ['N_'+str(x) for x in range(N_SYMBOLS)] + [str(x) for x in range(10)] + list(COMMON_CONSTANTS) \
                 + MATH_TOKENS + [chr(x+ord('a')) for x in range(26)]
    tgt_indexes = get_token_indexes(tgt_tokens)


    data = {}
    data['dict'] = {}
    data['dict']['src'] = src_indexes
    data['dict']['tgt'] = tgt_indexes
    data['idx2id'] = index

    src_idx_repr = []
    for src_sent in src:
        idx = [BOS]
        # idx += [src_indexes[s] if s in src_indexes else src_indexes[UNK_WORD] for s in src_sent]
        for s in src_sent:
            if s in src_indexes:
                idx.append(src_indexes[s])
            elif s.lower() in src_indexes:
                idx.append(src_indexes[s.lower()])
            else:
                idx.append(src_indexes[UNK_WORD])
        idx.append(EOS)
        src_idx_repr.append(idx)

    tgt_idx_repr = []
    for tgt_sent in tgt:
        idx = [BOS]
        idx += [tgt_indexes[s] if s in tgt_indexes else tgt_indexes[UNK_WORD] for s in tgt_sent]
        idx.append(EOS)
        tgt_idx_repr.append(idx)

    tgt_num_repr = []
    for nums in tgt_num:
        idx = [BOS]
        idx += [tgt_indexes[s] if s in tgt_indexes else tgt_indexes[UNK_WORD] for s in nums]
        idx.append(EOS)
        tgt_num_repr.append(idx)

    data['src'] = src_idx_repr
    data['tgt'] = tgt_idx_repr
    data['tgt_nums'] = tgt_num_repr  # constants and number symbols in targets
    data['ans'] = answers
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


def check_data(data, idx):
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
    parser.add_argument('--max-len', type=int, default=70)
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





