import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import argparse
import math
import time
from tqdm import tqdm

import config
# from src.gcl_model import Transformer, BiTransformer, Translator
from src.gcl_model2 import Transformer, BiTransformer, Translator
from dataset import TranslationDataset
# from transformer.Optim import ScheduledOptim
from transformer import Constants
import numpy as np


def prepare_dataloaders(data, opt):
    # ========= Preparing DataLoader =========#

    N = data['settings']['n_instances']
    train_len = int(N * opt.split)
    start_idx = int(opt.offset * N)
    print("Data split: {}".format(opt.split))
    print("Training starts at: {} out of {} instances".format(start_idx, N))

    if start_idx + train_len < N:
        train_src_insts = data['src'][start_idx: start_idx + train_len]
        train_tgt_insts = data['tgt'][start_idx: start_idx + train_len]
        train_tgt_nums = data['tgt_nums'][start_idx: start_idx + train_len]

        valid_src_insts = data['src'][start_idx + train_len:] + data['src'][:start_idx]
        valid_tgt_insts = data['tgt'][start_idx + train_len:] + data['tgt'][:start_idx]
        valid_tgt_nums = data['tgt_nums'][start_idx + train_len:] + data['tgt_nums'][:start_idx]
    else:
        valid_len = N - train_len
        valid_start_idx = start_idx - valid_len

        train_src_insts = data['src'][start_idx:] + data['src'][:valid_start_idx]
        train_tgt_insts = data['tgt'][start_idx:] + data['tgt'][:valid_start_idx]
        train_tgt_nums = data['tgt_nums'][start_idx:] + data['tgt_nums'][:valid_start_idx]

        valid_src_insts = data['src'][valid_start_idx: start_idx]
        valid_tgt_insts = data['tgt'][valid_start_idx: start_idx]
        valid_tgt_nums = data['tgt_nums'][valid_start_idx: start_idx]

    train_loader = torch.utils.data.DataLoader(
        TranslationDataset(
            src_word2idx=data['dict']['src'],
            tgt_word2idx=data['dict']['tgt'],
            src_insts=train_src_insts,
            tgt_insts=train_tgt_insts,
            tgt_nums=train_tgt_nums,
            permute_tgt=False),
        num_workers=2,
        batch_size=opt.batch_size,
        # collate_fn=collate_fn,
        shuffle=True)

    valid_loader = torch.utils.data.DataLoader(
        TranslationDataset(
            src_word2idx=data['dict']['src'],
            tgt_word2idx=data['dict']['tgt'],
            src_insts=valid_src_insts,
            tgt_insts=valid_tgt_insts,
            tgt_nums=valid_tgt_nums,
            permute_tgt=False),
        num_workers=2,
        batch_size=opt.batch_size)
        # collate_fn=collate_fn)

    if opt.bi:
        train_loader.collate_fn = train_loader.dataset.bidirectional_collate_fn
        valid_loader.collate_fn = valid_loader.dataset.bidirectional_collate_fn
    else:
        train_loader.collate_fn = train_loader.dataset.paired_collate_fn
        valid_loader.collate_fn = valid_loader.dataset.paired_collate_fn

    return train_loader, valid_loader


class Scheduler():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, n_current_steps=0, alpha=1e-7, update_steps=100):
        self._optimizer = optimizer
        self.n_current_steps = n_current_steps
        self.init_lr = alpha
        self.update_steps = update_steps

    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        return np.power(0.5, self.n_current_steps // self.update_steps)

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        # self.n_current_steps += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr


def cal_performance(pred, gold, smoothing=False, weight=None):
    ''' Apply label smoothing if needed '''

    # pred (batch*steps, vocab_size), gold(batch, steps)
    batch_size = gold.shape[0]
    n, m = pred.shape[0] // batch_size, gold.shape[1]
    if n > m:
        gold = torch.cat([gold, torch.zeros(batch_size, n - m).type(torch.cuda.LongTensor)], 1)
    elif n < m:
        pred = torch.cat([pred, torch.zeros(batch_size * (m - n), pred.shap[1]).type(torch.cuda.FloatTensor)], 0)

    loss = cal_loss(pred, gold, smoothing, weight)

    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    non_pad_mask = gold.ne(Constants.PAD)
    n_correct = pred.eq(gold)
    n_correct = n_correct.masked_select(non_pad_mask).sum().item()

    return loss, n_correct


def cal_loss(pred, gold, smoothing, weight):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(Constants.PAD)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later

    else:
        loss = F.cross_entropy(pred, gold, weight=weight, ignore_index=Constants.PAD, reduction='sum')

    return loss


def train_epoch(model, training_data, optimizer, device, smoothing, **kwargs):
    ''' Epoch operation in training phase'''

    model.train()
    optimizer.n_current_steps += 1

    model.decoder_lr.gcl.init_sequence(1)
    model.decoder_rl.gcl.init_sequence(1)
    model.decoder_lr.memory_ready = False
    model.decoder_rl.memory_ready = False

    total_loss = 0
    n_word_total = 0
    n_word_correct = 0

    weight = kwargs['weight']
    bidirectional = kwargs['bidirectional']

    for batch in tqdm(
            training_data, mininterval=2,
            desc='  - (Training)   ', leave=False):

        optimizer.zero_grad()
        # memory = model.encoder.gcl.memory
        # model.encoder.gcl.init_sequence(1)
        # model.encoder.gcl.memory = memory
        # model.encoder.gcl.gcl.memory = memory
        # for head in model.encoder.gcl.gcl.heads:
        #     head.memory = memory

        if bidirectional:
            # forward
            src_seq, src_pos, tgt_seq, tgt_seq_reversed, tgt_pos, *_ = map(lambda x: x.to(device), batch)
            gold_lr = tgt_seq[:, 1:]
            gold = gold_lr  # another name, for convenience
            gold_rl = tgt_seq_reversed[:, 1:]
            pred_lr, pred_rl = model(src_seq, src_pos, tgt_seq, tgt_seq_reversed, tgt_pos)
            # print(pred_lr.shape, pred_rl.shape, gold_lr.shape, gold_rl.shape)
            loss_lr, n_correct_lr = cal_performance(pred_lr, gold_lr, smoothing=smoothing, weight=weight)
            loss_rl, n_correct_rl = cal_performance(pred_rl, gold_rl, smoothing=smoothing, weight=weight)

            loss = 0.5 * (loss_lr + loss_rl)
            n_correct = 0.5 * (n_correct_lr + n_correct_rl)

        else:
            # forward
            src_seq, src_pos, tgt_seq, tgt_pos, *_ = map(lambda x: x.to(device), batch)
            gold = tgt_seq[:, 1:]
            pred = model(src_seq, src_pos, tgt_seq, tgt_pos)
            loss, n_correct = cal_performance(pred, gold, smoothing=smoothing, weight=weight)

        # backward
        loss.backward()
        # print(loss)

        # update parameters
        optimizer.step_and_update_lr()
        # print("get_lr_scale", optimizer._get_lr_scale())

        # note keeping
        total_loss += loss.item()

        non_pad_mask = gold.ne(Constants.PAD)
        n_word = non_pad_mask.sum().item()
        n_word_total += n_word
        n_word_correct += n_correct

    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    return loss_per_word, accuracy

def eval_epoch(model, validation_data, device, **kwargs):
    ''' Epoch operation in evaluation phase '''

    weight = kwargs['weight']
    bidirectional = kwargs['bidirectional']
    model.eval()

    total_loss = 0
    n_word_total = 0
    n_word_correct = 0

    with torch.no_grad():
        for batch in tqdm(
                validation_data, mininterval=2,
                desc='  - (Validation) ', leave=False):

            if bidirectional:
                src_seq, src_pos, tgt_seq, tgt_seq_reversed, tgt_pos, *_ = map(lambda x: x.to(device), batch)
                gold_lr = tgt_seq[:, 1:]
                gold = gold_lr  # another name, for convenience
                gold_rl = tgt_seq_reversed[:, 1:]
                pred_lr, pred_rl = model(src_seq, src_pos, tgt_seq, tgt_seq_reversed, tgt_pos)
                loss_lr, n_correct_lr = cal_performance(pred_lr, gold_lr, weight=weight)
                loss_rl, n_correct_rl = cal_performance(pred_rl, gold_rl, weight=weight)
                loss = 0.5 * (loss_lr + loss_rl)
                n_correct = 0.5 * (n_correct_lr + n_correct_rl)
            else:
                src_seq, src_pos, tgt_seq, tgt_pos, *_ = map(lambda x: x.to(device), batch)
                gold = tgt_seq[:, 1:]
                pred = model(src_seq, src_pos, tgt_seq, tgt_pos)
                # print(src_seq.shape, src_pos.shape, tgt_seq.shape, tgt_pos.shape);exit("Run time error")
                loss, n_correct = cal_performance(pred, gold, weight=weight)

            # note keeping
            total_loss += loss.item()

            non_pad_mask = gold.ne(Constants.PAD)
            n_word = non_pad_mask.sum().item()
            n_word_total += n_word
            n_word_correct += n_correct

    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    return loss_per_word, accuracy

def train_transformer(model, training_data, validation_data, optimizer, device, opt):
    ''' Start training '''

    log_train_file = None
    log_valid_file = None

    if opt.log:
        log_train_file = opt.log + '.train.log'
        log_valid_file = opt.log + '.valid.log'

        print('[Info] Training performance will be written to file: {} and {}'.format(
            log_train_file, log_valid_file))

        with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
            log_tf.write('epoch,loss,ppl,accuracy\n')
            log_vf.write('epoch,loss,ppl,accuracy\n')

    if opt.ops_idx is not None:
        # compute class weights
        tgt_vocab_size = len(training_data.dataset.tgt_word2idx)
        weight = torch.ones(tgt_vocab_size, dtype=torch.float)
        if torch.cuda.is_available():
            weight = weight.cuda()
        for item in opt.ops_idx:
            weight[item] *= 5  # 5 times the loss
    else:
        weight = None


    valid_accus = []
    for epoch_i in range(opt.epoch):
        if epoch_i > 50 and not model.decoder_lr.use_memory:
            model.decoder_lr.use_memory = True
            model.decoder_rl.use_memory = True
            print("use_memory set to True")
        print('[ Epoch', epoch_i, ']')

        start = time.time()
        train_loss, train_accu = train_epoch(
            model, training_data, optimizer, device, smoothing=opt.label_smoothing, weight=weight, bidirectional=opt.bi)
        print('  - (Training)   ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, '\
              'elapse: {elapse:3.3f} min'.format(
                  ppl=math.exp(min(train_loss, 100)), accu=100*train_accu,
                  elapse=(time.time()-start)/60))

        start = time.time()
        valid_loss, valid_accu = eval_epoch(model, validation_data, device, weight=weight, bidirectional=opt.bi)
        print('  - (Validation) ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, '\
                'elapse: {elapse:3.3f} min'.format(
                    ppl=math.exp(min(valid_loss, 100)), accu=100*valid_accu,
                    elapse=(time.time()-start)/60))

        valid_accus += [valid_accu]

        model_state_dict = model.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'memory_lr': model.decoder_lr.gcl.memory,
            'memory_rl': model.decoder_rl.gcl.memory,
            'settings': opt,
            'epoch': epoch_i}

        if opt.save_model:
            if opt.save_mode == 'all':
                model_name = opt.save_model + '_accu_{accu:3.3f}.chkpt'.format(accu=100*valid_accu)
                torch.save(checkpoint, model_name)
            elif opt.save_mode == 'best':
                model_name = opt.save_model + '.chkpt'
                if valid_accu >= max(valid_accus):
                    torch.save(checkpoint, model_name)
                    print('    - [Info] The checkpoint file has been updated.')
            elif opt.save_mode == 'interval':
                if (epoch_i + 1) % 50 == 0:
                    model_name = opt.save_model + '{}.chkpt'.format(epoch_i+1)
                    torch.save(checkpoint, model_name)
            else:
                model_name = opt.save_model + '.chkpt'
                torch.save(checkpoint, model_name)

        if log_train_file and log_valid_file:
            with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
                log_tf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                    epoch=epoch_i, loss=train_loss,
                    ppl=math.exp(min(train_loss, 100)), accu=100*train_accu))
                log_vf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                    epoch=epoch_i, loss=valid_loss,
                    ppl=math.exp(min(valid_loss, 100)), accu=100*valid_accu))

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-data', required=True)
    parser.add_argument('-split', type=float, default=0.8, help="portion for training")
    parser.add_argument('-offset', type=float, default=0, help="determin starting index of training set, for cross validation")

    parser.add_argument('-epoch', type=int, default=250)
    # parser.add_argument('-batch_size', type=int, default=128)
    parser.add_argument('-batch_size', type=int, default=64)
    parser.add_argument('-bi', action='store_true')

    parser.add_argument('-d_word_vec', type=int, default=300,
                        help="dimension of src text word vectors")
    parser.add_argument('-d_model', type=int, default=512,
                        help='size of encoder layer above embedding layer')
    parser.add_argument('-d_inner_hid', type=int, default=2048)
    parser.add_argument('-d_k', type=int, default=64)
    parser.add_argument('-d_v', type=int, default=64)

    parser.add_argument('-n_head', type=int, default=8)
    parser.add_argument('-n_layers', type=int, default=3)
    # parser.add_argument('-n_warmup_steps', type=int, default=4000)

    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-embs_share_weight', action='store_true')
    parser.add_argument('-proj_share_weight', action='store_true')

    parser.add_argument('-log', default=None)
    parser.add_argument('-save_model', default=None)
    parser.add_argument('-save_mode', type=str, choices=['all', 'best', 'interval', 'last'], default='last')

    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-label_smoothing', action='store_true')
    parser.add_argument('-load_model', type=str, default=None, help='load pretrained model')

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda
    # opt.d_word_vec = opt.d_model

    # ========= Loading Dataset =========#
    data = torch.load(opt.data)
    opt.max_token_seq_len = data['settings']['max_token_seq_len']

    training_data, validation_data = prepare_dataloaders(data, opt)

    opt.src_vocab_size = training_data.dataset.src_vocab_size
    opt.tgt_vocab_size = training_data.dataset.tgt_vocab_size

    ops_idx = [data['dict']['tgt'][s] for s in ('+', '-', '*', '/')]
    # ops_idx = None
    opt.ops_idx = ops_idx  # indexes of operators

    # ========= Preparing Model =========#
    if opt.embs_share_weight:
        assert training_data.dataset.src_word2idx == training_data.dataset.tgt_word2idx, \
            'The src/tgt word2idx table are different but asked to share word embedding.'

    print(opt)

    device = torch.device('cuda' if opt.cuda else 'cpu')
    if opt.bi:
        transformer = BiTransformer(
            opt.src_vocab_size,
            opt.tgt_vocab_size,
            opt.max_token_seq_len,
            tgt_emb_prj_weight_sharing=opt.proj_share_weight,
            emb_src_tgt_weight_sharing=opt.embs_share_weight,
            d_k=opt.d_k,
            d_v=opt.d_v,
            d_model=opt.d_model,
            d_word_vec=opt.d_word_vec,  # src word vector dimension
            d_inner=opt.d_inner_hid,
            n_layers=opt.n_layers,
            n_head=opt.n_head,
            dropout=opt.dropout,
            embedding_matrix=data['src_embeddings']).to(device)

    else:
        transformer = Transformer(
            opt.src_vocab_size,
            opt.tgt_vocab_size,
            opt.max_token_seq_len,
            tgt_emb_prj_weight_sharing=opt.proj_share_weight,
            emb_src_tgt_weight_sharing=opt.embs_share_weight,
            d_k=opt.d_k,
            d_v=opt.d_v,
            d_model=opt.d_model,
            d_word_vec=opt.d_word_vec,  # src word vector dimension
            d_inner=opt.d_inner_hid,
            n_layers=opt.n_layers,
            n_head=opt.n_head,
            dropout=opt.dropout,
            embedding_matrix=data['src_embeddings']).to(device)

    optimizer = Scheduler(
        optim.Adam(
            filter(lambda x: x.requires_grad, transformer.parameters()),
            betas=(0.9, 0.98), eps=1e-09),
            alpha=5e-5, n_current_steps=0)  # 5e-5

    if opt.load_model is not None:
        checkpoint = torch.load(opt.load_model)
        transformer.load_state_dict(checkpoint['model'])

    train_transformer(transformer, training_data, validation_data, optimizer, device, opt)


if __name__ == '__main__':
    main()
