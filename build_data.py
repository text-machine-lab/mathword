import torch
from torch.utils.data import Dataset, DataLoader
import json
from random import shuffle
import nltk
import re
import itertools
import numpy as np
import argparse

import config
from src.word2vec import build_vocab
from transformer.Constants import *


class MathWordData(Dataset):
    def __init__(self, datafile):
        self.data = json.load(open(datafile))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def equation_tokenize(equations):
    expr = ';'.join(equations)
    return list(expr)


def text_tokenize(question):
    words = nltk.word_tokenize(question)
    tokens = []
    for word in words:
        pattern0 = re.match(r'([a-zA-Z]+)(\d+)', word)
        pattern1 = re.match(r'(\d+)([a-zA-Z]+)', word)
        if pattern0:
            tokens.append(pattern0.group(1))
            tokens += list(pattern0.group(2))
        elif pattern1:
            tokens += list(pattern1.group(1))
            tokens.append(pattern1.group(2))
        elif re.search(r'[\d\.\+\-/,]', word):
            tokens += list(word)
        else:
            tokens.append(word)
    return tokens


def get_embedding_matrix(word_vectors=None):
    """
    special tokens:
    PAD_WORD = '<blank>'
    UNK_WORD = '<unk>'
    BOS_WORD = '<s>'
    EOS_WORD = '</s>'
    """
    n_special_toks = 4
    word_indexes = {}
    embedding_matrix = np.random.uniform(low=-0.5, high=0.5, size=(len(word_vectors) + n_special_toks, config.EMBEDDING_DIM))

    for index, word in enumerate(sorted(word_vectors.keys())):
        word_indexes[word] = index + n_special_toks
        embedding_vector = word_vectors.get(word, None)
        embedding_matrix[index + 4] = embedding_vector

    word_indexes[PAD_WORD] = 0
    word_indexes[UNK_WORD] = 1
    word_indexes[BOS_WORD] = 2
    word_indexes[EOS_WORD] = 3

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
    index = {}
    src_truncated = 0
    tgt_truncated = 0
    for i, d in enumerate(data):
        text_toks = text_tokenize(d['text'])
        if len(text_toks) > max_len:
            # print(text_toks)
            src_truncated += 1
            text_toks = text_toks[:max_len]

        equation_toks = equation_tokenize(d['equations'])
        if len(equation_toks) > max_len:
            # print(equation_toks)
            tgt_truncated += 1
            equation_toks = equation_toks[:max_len]

        src.append(text_toks)
        tgt.append(equation_toks)
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
    args = parser.parse_args()

    pretrained = not args.no_pretrained
    data = load_data(['/data2/ymeng/dolphin18k/formatted/eval_linear_auto_t1.json',
               '/data2/ymeng/dolphin18k/formatted/eval_linear_manual_t1.json'],
                     pretrained=pretrained, max_len=args.max_len)
    torch.save(data, 'models/data.pt')





