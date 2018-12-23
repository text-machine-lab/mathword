''' train with REINFORCE '''

import torch
import torch.utils.data
import torch.optim as optim
import torch.nn.functional as F
import argparse
from tqdm import tqdm
import json
import math
import time
import signal
import sys
import numpy as np
import copy

import config
from dataset import TranslationDataset
from src.model import Translator, print_grad
from src.check_answers import check_solution
from src.rougescore import rouge_n
from transformer.Constants import UNK_WORD
from train import Scheduler, cal_performance
from predict import reset_numbers


def get_equation_list(equation_str):
    s = equation_str.replace('</s>', '')
    equations = s.split(';')
    return equations


class Timeout():
    """Timeout class using ALARM signal."""

    class Timeout(Exception):
        pass

    def __init__(self, sec):
        self.sec = sec

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.raise_timeout)
        signal.alarm(self.sec)

    def __exit__(self, *args):
        signal.alarm(0)  # disable alarm

    def raise_timeout(self, *args):
        raise Timeout.Timeout()


def rouge_score(target, prediction, tgt_word2idx, alpha=0.5):
    # prediction = prediction.replace('</s>', '')
    # if 'n' in target and 'x' not in target:
    #     target = target.replace('n', 'x')
    # if 't' in target and 'x' not in target:
    #     target = target.replace('t', 'x')
    # if 'm' in target and 'y' not in target:
    #     target = target.replace('m', 'y')
    target = copy.copy(target)
    prediction = [x for x in prediction if tgt_word2idx['</s>'] != x]
    if len(target) > 2:
        target = target[1:-1]  # remove edge tokens

    if tgt_word2idx['n'] in target and tgt_word2idx['x'] not in target:
        target = [tgt_word2idx['x'] if s == tgt_word2idx['n'] else s for s in target]
    if tgt_word2idx['t'] in target and tgt_word2idx['x'] not in target:
        target = [tgt_word2idx['x'] if s == tgt_word2idx['t'] else s for s in target]
    if tgt_word2idx['m'] in target and tgt_word2idx['y'] not in target:
        target = [tgt_word2idx['y'] if s == tgt_word2idx['m'] else s for s in target]

    # print(target, prediction)
    # r_1 = rouge_n(target, [prediction], 1, alpha)
    r_2 = rouge_n(target, [prediction], 2, alpha)
    r_3 = rouge_n(target, [prediction], 3, alpha)

    # return (r_1 * r_2 * r_3)**(1 / 3)  # geometric mean
    return (r_2 + r_3) / 2

def sample_instances(scores, eps=0.01):
    """tensor input"""
    score_values = np.array([x.item() for x in scores.data])
    probs = np.exp(score_values) / (sum(np.exp(score_values)) + eps)
    cum_probs = probs.cumsum()
    u = np.random.rand(len(cum_probs), 1)
    choices = (u < cum_probs).argmax(axis=1)

    return choices


def teacher_train_batch(model, batch, optimizer, device):
    src_seq, src_pos, tgt_seq, tgt_seq_reversed, tgt_pos, *_ = map(lambda x: x.to(device), batch)
    gold_lr = tgt_seq[:, 1:]
    # gold = gold_lr  # another name, for convenience
    gold_rl = tgt_seq_reversed[:, 1:]
    pred_lr, pred_rl = model(src_seq, src_pos, tgt_seq, tgt_seq_reversed, tgt_pos)
    loss_lr, n_correct_lr = cal_performance(pred_lr, gold_lr)
    loss_rl, n_correct_rl = cal_performance(pred_rl, gold_rl)

    loss = loss_lr + loss_rl

    optimizer.zero_grad()
    # backward
    loss.backward()
    # print(loss)

    # update parameters
    optimizer.step_and_update_lr()


def main():
    '''Main Function'''

    parser = argparse.ArgumentParser(description='reinforcement training')

    parser.add_argument('-model', required=True,
                        help='Path to pretrained model .pt file')
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
    parser.add_argument('-save_model', default=None, help="model destination path")
    parser.add_argument('-beam_size', type=int, default=4,
                        help='Beam size')
    parser.add_argument('-batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('-n_best', type=int, default=4,
                        help="If verbose is set, will output the n_best decoded sentences")
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-epochs', type=int, default=10)
    parser.add_argument('-teacher_ratio', type=float, default=0.5, help="probability to allow teacher forcing")

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda
    opt.reset_num = True  # use numbers (not symbols) in output
    print(opt)

    # Prepare DataLoader
    preprocess_data = torch.load(opt.data)
    if opt.original_data is not None:
        formatted_data = json.load(open(opt.original_data))
        formatted_map = {}
        for d in formatted_data:
            formatted_map[d['id']] = d

    train_len = int(preprocess_data['settings']['n_instances'] * opt.split)

    data_loader = torch.utils.data.DataLoader(
        TranslationDataset(
            src_word2idx=preprocess_data['dict']['src'],
            tgt_word2idx=preprocess_data['dict']['tgt'],
            src_insts=preprocess_data['src'][:train_len],
            tgt_insts=preprocess_data['tgt'][:train_len],
            tgt_nums=preprocess_data['tgt_nums'][:train_len],
            permute_tgt=False),
        num_workers=0,
        batch_size=opt.batch_size)
        # collate_fn=collate_fn)
    # data_loader.collate_fn = data_loader.dataset.collate_fn
    data_loader.collate_fn = data_loader.dataset.bidirectional_collate_fn

    # tgt_insts = preprocess_data['tgt'][:train_len]
    # block_list = [preprocess_data['dict']['tgt'][UNK_WORD]]

    translator = Translator(opt)
    original_max_token_seq_len = translator.model_opt.max_token_seq_len
    translator.model.train()

    # set teacher forcing training optimizer
    optimizer_teacher = Scheduler(
        optim.Adam(
            filter(lambda x: x.requires_grad, translator.model.parameters()),
            betas=(0.9, 0.98), eps=1e-09),
        512, 4000, alpha=5e-5)
    # set reinforcement training optimizer
    optimizer_reinforce = Scheduler(
        optim.Adam(
            filter(lambda x: x.requires_grad, translator.model.parameters()),
            betas=(0.9, 0.98), eps=1e-09),
        512, 4000, alpha=1e-6)

    for epoch in range(opt.epochs):
        start = time.time()
        instance_idx = 0
        n_correct = 0
        total_loss = 0
        for batch in tqdm(data_loader, mininterval=2, desc='  - (Train)', leave=False):
            # batch: (*src_insts, *tgt_insts, *tgt_nums_insts)
            # print(batch[0]);sys.exit(1)
            translator.model_opt.max_token_seq_len = 32  # make training managable
            all_hyp_list, all_score_list = translator.translate_batch(batch[0], batch[1], block_list=[])

            choice = np.random.rand(1)
            if choice < opt.teacher_ratio:
                # teacher forceing training
                teacher_train_batch(translator.model, batch, optimizer_teacher, translator.device)

            # reinforcement training
            batch_loss, batch_n_correct = train_batch(all_hyp_list, all_score_list, translator, data_loader,
                                                    preprocess_data, formatted_map, instance_idx)
            optimizer_reinforce.zero_grad()
            batch_loss.backward()
            optimizer_reinforce.step_and_update_lr()

            total_loss += batch_loss.item()
            n_correct += batch_n_correct
            instance_idx += opt.batch_size

        # end of epoch
        train_acc = n_correct / train_len
        total_loss /= train_len
        print('  - (Training)   ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, ' \
              'elapse: {elapse:3.3f} min'.format(
            ppl=math.exp(min(total_loss, 100)), accu=100 * train_acc, elapse=(time.time() - start) / 60))

        model_state_dict = translator.model.state_dict()
        translator.model_opt.max_token_seq_len = original_max_token_seq_len
        checkpoint = {
            'model': model_state_dict,
            'settings': translator.model_opt,
            'epoch': epoch}
        model_name = opt.save_model + '.chkpt'
        torch.save(checkpoint, model_name)

def train_batch(all_hyp_list, all_score_list, translator, data_loader, preprocess_data, formatted_map, instance_idx):
    n_correct = 0
    batch_loss = []
    for i, idx_seqs in enumerate(all_hyp_list[0]):  # over instances in batch
        scores = all_score_list[0][i]
        choices = sample_instances(scores)

        if translator.opt.bi:  # bidirectional
            idx_seqs_reverse = all_hyp_list[1][i]
            scores_reverse = all_score_list[1][i]
            choices_reverse = sample_instances(scores_reverse)

        sampled_logprobs = []
        sampled_rewards = []
        for j, choice in enumerate(choices):  # over n_best results for an instance
            idx_seq = idx_seqs[choice]
            question_id = preprocess_data['idx2id'][instance_idx]

            pred_line = ''.join([data_loader.dataset.tgt_idx2word[idx] for idx in idx_seq])
            score = scores[choice] #/ len(idx_seq)

            if translator.opt.bi:
                idx_seq_reverse = idx_seqs_reverse[choices_reverse[j]]
                score_reverse = scores_reverse[choices_reverse[j]] #/ len(idx_seq_reverse)

                idx_seq_reverse = idx_seq_reverse[::-1]  # change to normal order
                pred_line_reverse = ''.join([data_loader.dataset.tgt_idx2word[idx] for idx in idx_seq_reverse])

            # src_idx_seq = data_loader.dataset[n]  # truth
            # src_text = ' '.join([data_loader.dataset.src_idx2word[idx] for idx in src_idx_seq])

            # src_text = reset_numbers(src_text, preprocess_data['numbers'][n])
            tgt_text = ';'.join(formatted_map[question_id]['equations'])

            pred_line = reset_numbers(pred_line, preprocess_data['numbers'][instance_idx])
            pred_equations = get_equation_list(pred_line)
            ans = preprocess_data['ans'][instance_idx]
            try:
                with Timeout(2):
                    point, solution = check_solution(ans, pred_equations)
            except Timeout.Timeout:
                point = 0
                solution = []
            if j == 0:  # use the first sample to calculate accuracty
                n_correct += point

            if point < 1:
                point = rouge_score(preprocess_data['tgt'][instance_idx], idx_seq, data_loader.dataset.tgt_word2idx)
                # if solution == []:
                #     point -= 0.5
            else:
                point *= 2

            if translator.opt.bi:
                pred_line_reverse = reset_numbers(pred_line_reverse, preprocess_data['numbers'][instance_idx])
                pred_equations_reverse = get_equation_list(pred_line_reverse)
                try:
                    with Timeout(2):
                        point_reverse, solution_reverse = check_solution(ans, pred_equations_reverse)
                except Timeout.Timeout:
                    point_reverse = 0
                    solution_reverse = []
                if point_reverse < 1:
                    point_reverse = rouge_score(preprocess_data['tgt'][instance_idx], idx_seq_reverse, data_loader.dataset.tgt_word2idx)
                    # if solution_reverse == []:
                    #     point_reverse -= 0.5
                else:
                    point_reverse *= 2
                # print(choices, choices_reverse)
                # print(pred_line, tgt_text, point, score)
                # print(pred_line_reverse, tgt_text, point_reverse, score_reverse, '\n')

                # print(score, score_reverse, type(prob))
                # score = score + score_reverse
                # point = 0.5 * (point + point_reverse)

            sampled_logprobs.append([score, score_reverse])
            sampled_rewards.append([point, point_reverse])
            # print(sampled_logprobs, sampled_rewards)

        instance_idx += 1

        # end of n_best loop (one instance)
        # baseline_reward = 1.0
        baseline_reward0 = sum([x[0] for x in sampled_rewards]) / len(sampled_rewards)
        baseline_reward1 = sum([x[1] for x in sampled_rewards]) / len(sampled_rewards)
        sampled_loss = []
        for log_prob, reward in zip(sampled_logprobs, sampled_rewards):
            sampled_loss.append(-log_prob[0] * (reward[0] - baseline_reward0))  # loss can be negative now
            sampled_loss.append(-log_prob[1] * (reward[1] - baseline_reward1))
            # sampled_loss.append(-log_prob * reward)

        loss = torch.sum(torch.stack(sampled_loss))
        batch_loss.append(loss)

    batch_loss = torch.sum(torch.stack(batch_loss))

    return batch_loss, n_correct


if __name__ == "__main__":
    main()
