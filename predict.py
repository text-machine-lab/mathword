''' Translate input text with trained model. '''

import torch
import torch.utils.data
import argparse
from tqdm import tqdm

import config
from dataset import collate_fn, TranslationDataset
from src.model import Translator
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
    parser.add_argument('-no_cuda', action='store_true')

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda

    # Prepare DataLoader
    preprocess_data = torch.load(opt.data)
    # preprocess_settings = preprocess_data['settings']
    # test_src_word_insts = read_instances_from_file(
    #     opt.src,
    #     preprocess_settings.max_word_seq_len,
    #     preprocess_settings.keep_case)
    # test_src_insts = convert_instance_to_idx_seq(
    #     test_src_word_insts, preprocess_data['dict']['src'])

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
    translator = Translator(opt)

    with open(opt.output, 'w') as f:
        n = 0
        for batch in tqdm(test_loader, mininterval=2, desc='  - (Test)', leave=False):
            all_hyp, all_scores = translator.translate_batch(*batch)
            for idx_seqs in all_hyp:
                for idx_seq in idx_seqs:
                    pred_line = ''.join([test_loader.dataset.tgt_idx2word[idx] for idx in idx_seq])

                    src_idx_seq = test_loader.dataset[n]  # truth
                    src_text = ' '.join([test_loader.dataset.src_idx2word[idx] for idx in src_idx_seq])
                    tgt_text = ''.join([test_loader.dataset.tgt_idx2word[idx] for idx in tgt_insts[n]])

                    f.write(str(n) + ': ')
                    f.write(src_text + '\n')
                    f.write(tgt_text + '\n')
                    f.write(pred_line + '\n\n')
                    n += 1
    print('[Info] Finished.')

if __name__ == "__main__":
    main()
