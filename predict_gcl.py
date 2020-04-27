''' Translate input text with trained model. '''

import torch
import torch.utils.data
import argparse
from tqdm import tqdm
import json

import config
from dataset import TranslationDataset
#from src.gcl_model import Translator
from src.gcl_model2 import Translator
from transformer.Constants import UNK_WORD

def main():
    '''Main Function'''

    parser = argparse.ArgumentParser(description='predict.py')

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
    parser.add_argument('-offset', type=float, default=0,
                        help="determin starting index of training set, for cross validation")
    parser.add_argument('-output', default='pred.json',
                        help="""Path to output the predictions (each line will be the decoded sequence""")
    parser.add_argument('-beam_size', type=int, default=10,
                        help='Beam size')
    parser.add_argument('-batch_size', type=int, default=64,
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


    N = preprocess_data['settings']['n_instances']
    train_len = int(N * opt.split)
    start_idx = int(opt.offset * N)  # start location of training data
    print("Data split: {}".format(opt.split))
    print("Training starts at: {} out of {} instances".format(start_idx, N))

    if start_idx + train_len < N:
        valid_src_insts = preprocess_data['src'][start_idx + train_len:] + preprocess_data['src'][:start_idx]
        valid_tgt_insts = preprocess_data['tgt'][start_idx + train_len:] + preprocess_data['tgt'][:start_idx]
    else:
        valid_len = N - train_len
        valid_start_idx = start_idx - valid_len

        valid_src_insts = preprocess_data['src'][valid_start_idx: start_idx]
        valid_tgt_insts = preprocess_data['tgt'][valid_start_idx: start_idx]

    test_loader = torch.utils.data.DataLoader(
        TranslationDataset(
            src_word2idx=preprocess_data['dict']['src'],
            tgt_word2idx=preprocess_data['dict']['tgt'],
            src_insts=valid_src_insts),
        num_workers=2,
        batch_size=opt.batch_size)
        # collate_fn=collate_fn)
    test_loader.collate_fn = test_loader.dataset.collate_fn

    tgt_insts = valid_tgt_insts
    block_list = [preprocess_data['dict']['tgt'][UNK_WORD]]

    translator = Translator(opt)
    translator.retrieve_only = True

    # translator = NTMTranslator(opt)
    translator.model.eval()
    translator.model.decoder_lr.use_memory = True
    translator.model.decoder_rl.use_memory = True
    translator.model.decoder_lr.memory_ready = True
    translator.model.decoder_rl.memory_ready = True
    print(translator.model.decoder_lr.gcl.gcl.heads[0].memory.content[0])
    print("use_memory", translator.model.decoder_lr.use_memory)
    output = []
    n = 0
    for batch in tqdm(test_loader, mininterval=2, desc='  - (Test)', leave=False):
        # translator.model.encoder.gcl.init_sequence(1)
        with torch.no_grad():
            all_hyp_list, all_score_list = translator.translate_batch(*batch, block_list=block_list)
        for i, idx_seqs in enumerate(all_hyp_list[0]):  # loop over instances in batch
            scores = all_score_list[0][i]
            if translator.opt.bi:  # bidirectional
                idx_seqs_reverse = all_hyp_list[1][i]
                scores_reverse = all_score_list[1][i]

            for j, idx_seq in enumerate(idx_seqs):  # loop over n_best results
                d = {}
                question_id = preprocess_data['idx2id'][(n + train_len + start_idx) % N]

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
                    src_text = reset_numbers(src_text, preprocess_data['numbers'][(n + train_len + start_idx) % N])
                    # tgt_text = reset_numbers(tgt_text, preprocess_data['numbers'][n + train_len])
                    tgt_text = ';'.join(formmated_map[question_id]['equations'])

                    pred_line = reset_numbers(pred_line, preprocess_data['numbers'][(n + train_len + start_idx) % N], try_similar=True)
                    if translator.opt.bi:
                        pred_line_reverse = reset_numbers(pred_line_reverse, preprocess_data['numbers'][(n + train_len + start_idx) % N], try_similar=True)
                        # print(pred_line, tgt_text)
                        # print(pred_line_reverse, tgt_text, '\n')

                d['question'] = src_text
                d['ans'] = preprocess_data['ans'][(n + train_len + start_idx) % N]
                d['id'] = question_id
                d['equation'] = tgt_text
                d['pred'] = (pred_line.replace('</s>', ''), round(score.item(), 3) )
                if translator.opt.bi:
                    d['pred_2'] = (pred_line_reverse.replace('</s>', ''), round(score_reverse.item(), 3))

                output.append(d)
            n += 1

    with open(opt.output, 'w') as f:
        json.dump(output, f, indent=2)
    print('[Info] Finished.')


def reset_numbers(text, number_dict, try_similar=False):
    for k, v in number_dict.items():
        if not try_similar:
            text = text.replace(k, v)
        else:
            N_k = 'N' + k[1:]
            M_k = 'M' + k[1:]
            F_k = 'F' + k[1:]
            text = text.replace(N_k, v)
            text = text.replace(M_k, v)
            text = text.replace(F_k, v)

    text = text.replace('--', '+')
    return text

if __name__ == "__main__":
    main()
