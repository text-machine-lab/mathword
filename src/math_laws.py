import torch
import random
import re
import numpy as np
import config
from transformer import Constants

COMMON_CONSTANTS = set(['0.5', '1', '2', '12', '10', '100', '1000', '60', '180', '360', '3.14', '3.1416'])

def paren(s, token_prefix=''):
    return token_prefix + '(' + s + token_prefix + ')'

def permute(equations_str, token_prefix=''):
    T = token_prefix
    NUMBER = r'({}\d+(\.\d+)?|{}N_\d|{}x|{}y|{}z)'.format(T, T, T, T, T)

    equations = equations_str.split(T+';')
    random.shuffle(equations)  # order of equations does not matter

    new_equations = []
    for equation in equations:
        sides= equation.split(T+'=')

        if len(sides) < 2:
            new_equations.append(equation)
            continue

        random.shuffle(sides)  # order of sides does not matter
        left, right = sides

        # additions
        lm = re.match(r'{}{}\+{}$'.format(NUMBER, T, NUMBER), left)
        if lm:
            choice = np.random.randint(2)
            if choice == 0:
                left = lm.group(3) + T+'+' + lm.group(1)

        # divisions
        lm = re.match(r'{}{}/{}$'.format(NUMBER, T, NUMBER), left)
        if lm:
            choice = np.random.randint(2)
            if choice == 0:
                left = lm.group(1)
                if re.match(r'({}{}\*)?{}$'.format(NUMBER, T, NUMBER), right) or \
                        re.match(r'({}{}/)?{}$'.format(NUMBER, T, NUMBER), right):
                    right = right + T+'*' + lm.group(3)
                else:
                    right = paren(right, token_prefix=token_prefix) + T+'*' + lm.group(3)

        lm = re.match(r'{}{}/{}$'.format(NUMBER, T, NUMBER), left)
        rm = re.match(r'{}{}/{}$'.format(NUMBER, T, NUMBER), right)
        if lm and rm:
            # print(equation, lm.group(), rm.group(), lm.group(3), rm.group(3))
            choice = np.random.randint(4)
            if choice == 0:
                left = lm.group(3) + T+'/' + lm.group(1)
                right = rm.group(3) + T+'/' + rm.group(1)
            elif choice == 1:
                left = lm.group(1)
                right = right + T+'*' + lm.group(3)
            elif choice == 2:
                left = left + T+'*' + rm.group(3)
                right = rm.group(1)

        # multiplications
        lm = re.match(r'{}{}\*{}$'.format(NUMBER, T, NUMBER), left)
        if lm:
            choice = np.random.randint(2)
            if choice == 0:
                left = lm.group(1)
                if re.match(r'({}{}\*)?{}$'.format(NUMBER, T, NUMBER), right) or \
                        re.match(r'({}{}/)?{}$'.format(NUMBER, T, NUMBER), right):
                    right = right + T+'/' + lm.group(3)
                else:
                    right = paren(right, token_prefix=token_prefix) + T+'/' + lm.group(3)

        equation = (T+'=').join([left, right])
        new_equations.append(equation)

    return (T+';').join(new_equations)

def permute_tgt(tgt_batch, tgt_tok2idx_dict=None, token_prefix='__'):
    packed = True if tgt_batch[0][0] == Constants.BOS else False
    # print(packed)
    # print(tgt_batch)
    if tgt_tok2idx_dict is None:
        d = torch.load('models/data.pt')
        tgt_tok2idx_dict = d['dict']['tgt']

    tgt_idx2tok_dict = {}
    for k, v in tgt_tok2idx_dict.items():
        tgt_idx2tok_dict[v] = k

    new_seq = []
    new_tgt_nums = []
    for seq in tgt_batch:
        if packed:
            seq = seq[1:-1]
        equations = ''.join([token_prefix+tgt_idx2tok_dict[x] for x in seq])
        equations = permute(equations, token_prefix=token_prefix)
        # print(equations)
        try:
            equation_tokens = equations.split(token_prefix)
        except AttributeError:
            print("equation_tokens", equation_tokens)
            raise AttributeError
        # equation_seq = [tgt_tok2idx_dict[x.replace(token_prefix, '')] for x in equation_tokens if x != '']
        equation_seq = []
        tgt_nums = []
        for tok in equation_tokens:
            tok = tok.replace(token_prefix, '')
            if tok == '':
                continue
            tok_idx = tgt_tok2idx_dict[tok]
            equation_seq.append(tok_idx)
            if tok[0] == 'N' or tok in COMMON_CONSTANTS:
                tgt_nums.append(tok_idx)

        if packed:
            equation_seq = [Constants.BOS] + equation_seq + [Constants.EOS]
        new_seq.append(equation_seq)
        new_tgt_nums.append(tgt_nums)

    return new_seq, new_tgt_nums









