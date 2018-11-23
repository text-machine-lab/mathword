''' Translate input text with trained model. '''

import torch
import torch.utils.data
import argparse
from tqdm import tqdm
import json

import config
from dataset import collate_fn, TranslationDataset
from src.model import Translator
from transformer.Constants import UNK_WORD
from preprocess import read_instances_from_file, convert_instance_to_idx_seq

def main():
    '''Main Function'''

    parser = argparse.ArgumentParser(description='translate.py')

    parser.add_argument('-model', required=True,
                        help='Path to model .pt file')
    # parser.add_argument('-src', required=True,
    #                     help='Source sequence to decode (one line per sequence)')
    parser.add_argument('-data', required=True,
                        help='preprocessed data file')
    parser.add_argument('-original_data', default=config.FORMATTED_DATA,
                        help='original data showing original text and equations')
    parser.add_argument('-vocab', default=None,
                        help='data file for vocabulary. if not specified (default), use the one in -data')
    parser.add_argument('-split', type=float, default=0.8,
                        help='proprotion of training data. the rest is test data.')
    parser.add_argument('-output', default='pred.txt',
                        help="""Path to output the predictions (each line will
                        be the decoded sequence""")
    parser.add_argument('-beam_size', type=int, default=5,
                        help='Beam size')
    parser.add_argument('-batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('-n_best', type=int, default=1,
                        help="""If verbose is set, will output the n_best
                        decoded sentences""")
    parser.add_argument('-reset_num', default=False, action='store_true', help='replace number symbols with real numbers')
    parser.add_argument('-no_cuda', action='store_true')

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda
    print(opt)

    # Prepare DataLoader
    preprocess_data = torch.load(opt.data)
    if opt.original_data is not None:
        formmated_data = json.load(open(opt.original_data))
        formmated_map = {}
        for d in formmated_data:
            formmated_map[d['id']] = d


    train_len = int(preprocess_data['settings']['n_instances'] * opt.split)

    test_loader = torch.utils.data.DataLoader(
        TranslationDataset(
            src_word2idx=preprocess_data['dict']['src'],
            tgt_word2idx=preprocess_data['dict']['tgt'],
            src_insts=preprocess_data['src'][train_len:]),
        num_workers=2,
        batch_size=opt.batch_size,
        collate_fn=collate_fn)

    tgt_insts = preprocess_data['tgt'][train_len:]
    unk_idx = preprocess_data['dict']['tgt'][UNK_WORD]

    translator = Translator(opt)

    with open(opt.output, 'w') as f:
        n = 0
        for batch in tqdm(test_loader, mininterval=2, desc='  - (Test)', leave=False):
            all_hyp_list, all_score_list = translator.translate_batch(*batch, block=unk_idx)
            for i, idx_seqs in enumerate(all_hyp_list[0]):
                scores = all_score_list[0][i]
                if translator.opt.bi:  # bidirectional
                    idx_seqs_reverse = all_hyp_list[1][i]
                    scores_reverse = all_score_list[1][i]

                for j, idx_seq in enumerate(idx_seqs):
                    question_id = preprocess_data['idx2id'][n+train_len]

                    pred_line = ''.join([test_loader.dataset.tgt_idx2word[idx] for idx in idx_seq])
                    score = scores[j]
                    if translator.opt.bi:
                        idx_seq_reverse = idx_seqs_reverse[j]
                        score_reverse = scores_reverse[j]
                        idx_seq_reverse.reverse()
                        pred_line_reverse = ''.join([test_loader.dataset.tgt_idx2word[idx] for idx in idx_seq_reverse])


                    src_idx_seq = test_loader.dataset[n]  # truth
                    src_text = ' '.join([test_loader.dataset.src_idx2word[idx] for idx in src_idx_seq])
                    tgt_text = ''.join([test_loader.dataset.tgt_idx2word[idx] for idx in tgt_insts[n]])
                    if opt.reset_num:
                        src_text = reset_numbers(src_text, preprocess_data['numbers'][n + train_len])
                        # tgt_text = reset_numbers(tgt_text, preprocess_data['numbers'][n + train_len])
                        tgt_text = ';'.join(formmated_map[question_id]['equations'])

                        pred_line = reset_numbers(pred_line, preprocess_data['numbers'][n + train_len])
                        if translator.opt.bi:
                            pred_line_reverse = reset_numbers(pred_line_reverse, preprocess_data['numbers'][n + train_len])

                    f.write(str(n) + ': ')
                    f.write(src_text + '\n')
                    f.write(tgt_text + '\n')
                    f.write(question_id + '\n')
                    f.write(pred_line.replace('</s>', '') + '\t' + str(score.item()) + '\n')
                    f.write(pred_line_reverse.replace('</s>', '') + '\t' + str(score_reverse.item()) + '\n\n')
                    n += 1

    print('[Info] Finished.')

def reset_numbers(text, number_dict):
    for k, v in number_dict.items():
        text = text.replace(k, v)
    return text

if __name__ == "__main__":
    main()
