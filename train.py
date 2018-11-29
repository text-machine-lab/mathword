import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import argparse
import math
import time
from tqdm import tqdm
import numpy as np
import sys

import config
from src.model import Transformer, BiTransformer, Translator
from dataset import TranslationDataset # paired_collate_fn, bidirectional_collate_fn
from transformer.Optim import ScheduledOptim
from transformer import Constants


def prepare_dataloaders(data, opt):
    # ========= Preparing DataLoader =========#

    train_len = int(data['settings']['n_instances'] * opt.split)
    train_loader = torch.utils.data.DataLoader(
        TranslationDataset(
            src_word2idx=data['dict']['src'],
            tgt_word2idx=data['dict']['tgt'],
            src_insts=data['src'][:train_len],
            tgt_insts=data['tgt'][:train_len],
            tgt_nums=data['tgt_nums'][:train_len],
            permute_tgt=False),
        num_workers=2,
        batch_size=opt.batch_size,
        # collate_fn=collate_fn,
        shuffle=True)

    valid_loader = torch.utils.data.DataLoader(
        TranslationDataset(
            src_word2idx=data['dict']['src'],
            tgt_word2idx=data['dict']['tgt'],
            src_insts=data['src'][train_len:],
            tgt_insts=data['tgt'][train_len:],
            tgt_nums=data['tgt_nums'][train_len:],
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


class Scheduler(ScheduledOptim):
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, d_model, n_warmup_steps, n_current_steps=0):
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = n_current_steps
        # self.init_lr = np.power(d_model, -0.5)  # this is from original code
        self.init_lr = 5e-5 * np.power(n_warmup_steps, 0.5)

def cal_performance(pred, gold, smoothing=False, weight=None, training_opt=None):
    ''' Apply label smoothing if needed '''

    batch_size = gold.size()[0]
    seq_len = gold.size()[1]
    # if training_opt is not None:
    #     # skip the first token for loss computation (allowing flexible equations)
    #     pred = pred.view(batch_size, -1, pred.size(-1))
    #     if pred.size()[1] > 3:
    #         pred_ = pred[:,1:,:].contiguous().view(-1, pred.size(-1))
    #         gold_ = gold[:,1:]
    #     else:
    #         pred_ = pred.contiguous().view(-1, pred.size(-1))
    #         gold_ = gold
    #     loss = cal_loss(pred_, gold_, smoothing, weight)
    #     pred = pred.contiguous().view(-1, pred.size(-1))
    #
    # else:
    loss = cal_loss(pred, gold, smoothing, weight, training_opt=training_opt)
    pred = pred.max(1)[1]

    gold = gold.contiguous().view(-1)
    non_pad_mask = gold.ne(Constants.PAD)
    n_correct = pred.eq(gold)
    n_correct = n_correct.masked_select(non_pad_mask).sum().item()

    pred = pred.view(batch_size, seq_len)
    if training_opt and training_opt.bi:
        gold = gold.view(batch_size, seq_len)
        alignment_penalty = cal_alignment_penalty(pred, gold, training_opt.symbols_idx)
        alignment_penalty = torch.FloatTensor(alignment_penalty)
        if torch.cuda.is_available():
            alignment_penalty = alignment_penalty.cuda()
        loss = torch.dot(alignment_penalty, loss)

    return loss, n_correct


def cal_alignment_penalty(pred_idx, gold_idx, symbols_idx):
    def compute_distance(list0, list1):
        length0 = len(list0)
        length1 = len(list1)
        penalty = abs(length0 - length1)
        n = min(length0, length1)
        for i in range(n):
            if list0[i] != list1[i]:
                penalty += 1
        return penalty

    assert pred_idx.size() == gold_idx.size()
    distances = []
    for pred, gold in zip(pred_idx, gold_idx):
        pred_symbols = [x.item() for x in pred if x.item() in symbols_idx]
        gold_symbols = [x.item() for x in gold if x.item() in symbols_idx]
        distance = min(5., compute_distance(pred_symbols, gold_symbols))
        distances.append(distance)

    distances = np.exp(0.5 * np.array(distances))
    return distances

def cal_loss(pred, gold, smoothing, weight, training_opt=None):
    ''' Calculate cross entropy loss, apply label smoothing if needed.
       pred size = [batch*seq_len, voc_size]
       gold size = [batch, seq_len]
    '''

    batch_size = gold.size()[0]
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
        if training_opt and training_opt.bi:
            # loss size = [batch*seq_len]
            loss = F.cross_entropy(pred, gold, weight=weight, ignore_index=Constants.PAD, reduction='none')
            loss = loss.view(batch_size, -1)
            loss = torch.sum(loss, dim=1)  # get size [batch]
        else:
            # loss is a tensor of single variable
            loss = F.cross_entropy(pred, gold, weight=weight, ignore_index=Constants.PAD, reduction='sum')

    return loss


def train_epoch(model, training_data, optimizer, device, smoothing, **kwargs):
    ''' Epoch operation in training phase'''

    model.train()

    total_loss = 0
    n_word_total = 0
    n_word_correct = 0

    weight = kwargs['weight']
    bidirectional = kwargs['bidirectional']
    training_opt = kwargs['training_opt']

    for batch in tqdm(
            training_data, mininterval=2,
            desc='  - (Training)   ', leave=False):

        optimizer.zero_grad()

        data_batch = map(lambda x: x.to(device), batch)
        if bidirectional:
            # forward
            src_seq, src_pos, tgt_seq, tgt_seq_reversed, tgt_pos, tgt_nums, tgt_nums_reversed, _ = data_batch
            gold_lr = tgt_seq[:, 1:]
            gold = gold_lr  # another name, for convenience
            gold_rl = tgt_seq_reversed[:, 1:]
            # gold_nums_lr = tgt_nums[:, 1:]
            # gold_nums_rl = tgt_nums_reversed[:, 1:]

            pred_lr, pred_rl = model(src_seq, src_pos, tgt_seq, tgt_seq_reversed, tgt_pos)
            loss_lr, n_correct_lr = cal_performance(pred_lr, gold_lr, smoothing=smoothing,
                                                    weight=weight, training_opt=None)  # training_opt controls loss

            loss_rl, n_correct_rl = cal_performance(pred_rl, gold_rl, smoothing=smoothing,
                                                    weight=weight, training_opt=None)
            # loss_nums_lr, _ = cal_performance(pred_nums_lr, gold_nums_lr)
            # loss_nums_rl, _ = cal_performance(pred_nums_rl, gold_nums_rl)

            # print(alignment_penalty_lr, alignment_penalty_rl)
            # sys.exit(1)

            loss = 0.5 * (loss_lr + loss_rl)
            n_correct = 0.5 * (n_correct_lr + n_correct_rl)

            if gold.size()[0] <= 16:  # small batches
                loss_lr.backward(retain_graph=True)
                loss_rl.backward()

            elif np.random.randint(2) == 0:  # make a choice for decoder to update
                loss_lr.backward()
            else:
                loss_rl.backward()

        else:
            # forward
            src_seq, src_pos, tgt_seq, tgt_pos, tgt_nums, _ = data_batch
            gold = tgt_seq[:, 1:]
            pred = model(src_seq, src_pos, tgt_seq, tgt_pos)
            loss, n_correct = cal_performance(pred, gold, smoothing=smoothing, weight=weight)
            # backward
            loss.backward()

        # update parameters
        optimizer.step_and_update_lr()

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

            data_batch = map(lambda x: x.to(device), batch)
            if bidirectional:
                src_seq, src_pos, tgt_seq, tgt_seq_reversed, tgt_pos, tgt_nums, tgt_nums_reversed, _ = data_batch
                gold_lr = tgt_seq[:, 1:]
                gold = gold_lr  # another name, for convenience
                gold_rl = tgt_seq_reversed[:, 1:]
                pred_lr, pred_rl = model(src_seq, src_pos, tgt_seq, tgt_seq_reversed, tgt_pos)

                loss_lr, n_correct_lr = cal_performance(pred_lr, gold_lr, weight=weight)
                loss_rl, n_correct_rl = cal_performance(pred_rl, gold_rl, weight=weight)
                loss = 0.5 * (loss_lr + loss_rl)
                n_correct = 0.5 * (n_correct_lr + n_correct_rl)
            else:
                src_seq, src_pos, tgt_seq, tgt_pos, tgt_nums, _ = data_batch
                gold = tgt_seq[:, 1:]
                pred = model(src_seq, src_pos, tgt_seq, tgt_pos)
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
        print('[ Epoch', epoch_i, ']')

        start = time.time()
        train_loss, train_accu = train_epoch(model, training_data, optimizer, device, smoothing=opt.label_smoothing,
                                             weight=weight, bidirectional=opt.bi, training_opt=opt)
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

    parser.add_argument('-epoch', type=int, default=10)
    parser.add_argument('-batch_size', type=int, default=32)
    parser.add_argument('-bi', action='store_true')

    parser.add_argument('-d_word_vec', type=int, default=300,
                        help="dimension of src text word vectors")
    parser.add_argument('-d_model', type=int, default=512,
                        help='size of encoder layer above embedding layer')
    parser.add_argument('-d_inner_hid', type=int, default=2048)
    parser.add_argument('-d_k', type=int, default=64)
    parser.add_argument('-d_v', type=int, default=64)

    parser.add_argument('-n_head', type=int, default=8)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-n_warmup_steps', type=int, default=1000)

    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-embs_share_weight', action='store_true')
    parser.add_argument('-proj_share_weight', action='store_true')

    parser.add_argument('-log', default=None)
    parser.add_argument('-save_model', default=None)
    parser.add_argument('-save_mode', type=str, choices=['all', 'best', 'last'], default='best')

    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-label_smoothing', action='store_true')
    parser.add_argument('-load_model', type=str, default=None, help='load pretrained model')
    parser.add_argument('-step', type=int, default=0, help='set initial step (related to learning rate)')

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda
    # opt.d_word_vec = opt.d_model

    # ========= Loading Dataset =========#
    data = torch.load(opt.data)
    opt.max_token_seq_len = data['settings']['max_token_seq_len']

    training_data, validation_data = prepare_dataloaders(data, opt)

    opt.src_vocab_size = training_data.dataset.src_vocab_size
    opt.tgt_vocab_size = training_data.dataset.tgt_vocab_size

    opt.ops_idx = [data['dict']['tgt'][s] for s in ('+', '-', '*', '/')]  # indexes of operators
    opt.symbols_idx = [v for k, v in data['dict']['tgt'].items() if k[0] == 'N']

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
        opt.d_model, opt.n_warmup_steps, n_current_steps=opt.step)

    if opt.load_model is not None:
        checkpoint = torch.load(opt.load_model)
        transformer.load_state_dict(checkpoint['model'])

    train_transformer(transformer, training_data, validation_data, optimizer, device, opt)


if __name__ == '__main__':
    main()