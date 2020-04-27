import torch
import torch.nn as nn
import transformer.Constants as Constants
from transformer.Models import Encoder as OriginalEncoder
from transformer.Models import Decoder as OriginalDecoder
from transformer.Models import get_sinusoid_encoding_table, EncoderLayer, DecoderLayer, get_non_pad_mask, get_attn_key_pad_mask, get_subsequent_mask
from transformer.Beam import Beam
from transformer.Translator import Translator as OriginalTranslator
import torch.nn.functional as F
import copy
import sys
import time

from .gcl import EncapsulatedGCL #, EncapsulatedGCLDual


def print_grad(name):
    def hook(grad):
        print('{} grad: {}'.format(name, grad))
    return hook

class Encoder(OriginalEncoder):
    """Modify original encoder to allow pretrained embeddings"""
    def __init__(
            self,
            n_src_vocab, len_max_seq, d_word_vec,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1, embedding_matrix=None):

        nn.Module.__init__(self)

        n_position = len_max_seq + 1

        if embedding_matrix is None:
            self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=Constants.PAD)
        else:
            self.src_word_emb = nn.Embedding.from_pretrained(
                torch.from_numpy(embedding_matrix).type(torch.cuda.FloatTensor),freeze=True)

            print("set pretrained word embeddings, size {}".format(self.src_word_emb.weight.size()))

        self.src_word_enc = nn.Linear(d_word_vec, d_model)  # need this because d_word_vec != d_model
        self.src_enc_dropout = nn.Dropout(p=0.5)  # token embedding dropout

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_model, padding_idx=0),
            freeze=True)
        print("set positions encoder, size{}".format(self.position_enc.weight.size()))

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, src_seq, src_pos, return_attns=False):

        enc_slf_attn_list = []

        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq)
        non_pad_mask = get_non_pad_mask(src_seq)

        # -- Forward
        src_word_emb = self.src_enc_dropout(self.src_word_emb(src_seq))
        enc_output = F.tanh(self.src_word_enc(src_word_emb)) + self.position_enc(src_pos)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,


class Decoder(OriginalDecoder):
    def __init__(self, n_tgt_vocab, len_max_seq,
            d_word_vec, d_model, d_inner,
            n_layers, n_head, d_k, d_v,
            dropout=0.1):

        nn.Module.__init__(self)
        n_position = len_max_seq + 1

        self.tgt_word_emb = nn.Embedding(
            n_tgt_vocab, d_word_vec, padding_idx=Constants.PAD)

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_word_vec, padding_idx=0),
            freeze=True)

        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

        self.last_layer = DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)

        self.use_memory = False
        self.memory_ready = False

        # # num_inputs, num_outputs, controller_size, controller_layers, num_heads, N, M, K, controller_type = 'MLP')
        # num_outputs = self.tgt_word_prj.weight.shape[-1]
        self.gcl = EncapsulatedGCL(2 * 72, 2 * 72, 2 * 72, 0, 1, N=128, M=2 * 72, K=d_model,
                                   controller_type='simple')
        self.gcl.init_sequence(1)
        # self.tuner = nn.Bilinear(d_model, d_model, 1)
        self.lstm = nn.LSTM(d_model, d_model, bidirectional=False, batch_first=True)
        self.bilinear = nn.Bilinear(d_model, d_model, 1)

    def forward(self, tgt_seq, tgt_pos, src_seq, enc_output, return_attns=False, retrieve_only=False):

        def forward_seq(tgt_seq, tgt_pos, src_seq, enc_output, return_attns=return_attns):

            dec_slf_attn_list, dec_enc_attn_list = [], []

            # -- Prepare masks
            non_pad_mask = get_non_pad_mask(tgt_seq)

            slf_attn_mask_subseq = get_subsequent_mask(tgt_seq).type(torch.cuda.LongTensor)
            slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=tgt_seq, seq_q=tgt_seq).type(torch.cuda.LongTensor)
            slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)

            dec_enc_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=tgt_seq)

            # -- Forward
            dec_output = self.tgt_word_emb(tgt_seq) + self.position_enc(tgt_pos)

            for dec_layer in self.layer_stack:
                dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                    dec_output, enc_output,
                    non_pad_mask=non_pad_mask,
                    slf_attn_mask=slf_attn_mask,
                    dec_enc_attn_mask=dec_enc_attn_mask)

                if return_attns:
                    dec_slf_attn_list += [dec_slf_attn]
                    dec_enc_attn_list += [dec_enc_attn]

            # if return_attns:
            #     return dec_output, dec_slf_attn_list, dec_enc_attn_list
            return dec_output,

        if not retrieve_only:
            dec_output, *_ = forward_seq(tgt_seq, tgt_pos, src_seq, enc_output, return_attns=False)
        if not self.use_memory:
            return dec_output,

        # for gcl
        k = F.max_pool1d(enc_output.permute(0, 2, 1), enc_output.shape[-2]).squeeze(-1)
        batch_size = k.shape[0]
        d_model = k.shape[-1]

        if not retrieve_only:
            x0 = tgt_seq.type(torch.cuda.FloatTensor)
            x1 = tgt_pos.type(torch.cuda.FloatTensor)

            n_positions = x1.shape[1]
            n_paddings = 72 - n_positions
            if n_paddings > 0:
                x0 = torch.cat([x0, torch.zeros(x0.shape[0], n_paddings).type(torch.cuda.FloatTensor)], -1)
                x1 = torch.cat([x1, torch.zeros(x1.shape[0], n_paddings).type(torch.cuda.FloatTensor)], -1)
            x = torch.cat([x0, x1], -1)
        else:
            x = torch.zeros(batch_size, 2*72).type(torch.cuda.FloatTensor)

        gcl_output = self.gcl(k.unsqueeze(0).detach(), x.unsqueeze(0), bidirectional=False, save_attn=False)
        retrieval = gcl_output.round().type(torch.cuda.LongTensor).view(batch_size, -1)

        if not retrieve_only:
            if not self.memory_ready:
                if torch.max(self.gcl.memory.content[0, -1, :]) > 0:  # last slot
                    self.memory_ready = True
            # print(self.memory_ready)
            if not self.memory_ready:
                return dec_output,

        pos_part = retrieval[:, -72:]
        max_len = torch.max(pos_part)
        # print(max_len);sys.exit(1)
        retrieval_seq = retrieval[:, :max_len]
        retrieval_pos = retrieval[:, 72:72 + max_len]

        if retrieve_only:
            return retrieval_seq, retrieval_pos

        if torch.max(retrieval_pos[0]) == 0 and batch_size > 1:  # empty memory
            retrieval_seq[0] = retrieval_seq[1]
            retrieval_pos[0] = retrieval_pos[1]
        retrieval_dec_output, *_ = forward_seq(retrieval_seq, retrieval_pos, src_seq, enc_output, return_attns=False)

        dec_output_lstm, _ = self.lstm(dec_output)
        retrieval_dec_output_lstm, _ = self.lstm(retrieval_dec_output)
        dec_output_maxpool = F.max_pool1d(dec_output_lstm.permute(0, 2, 1), dec_output_lstm.shape[-2]).squeeze(-1)
        retrieval_output_maxpool = F.max_pool1d(retrieval_dec_output_lstm.permute(0, 2, 1),
                                                retrieval_dec_output_lstm.shape[-2]).squeeze(-1)
        gates = F.sigmoid(self.bilinear(dec_output_maxpool, retrieval_output_maxpool)).unsqueeze(-1)

        n, m = dec_output.shape[1], retrieval_dec_output.shape[1]
        if n > m:
            retrieval_dec_output = torch.cat(
                [retrieval_dec_output, torch.zeros(batch_size, n - m, d_model).type(torch.cuda.FloatTensor)], 1)
        elif n < m:  # retrieved tgt_seq is longer
            dec_output = torch.cat([dec_output, torch.zeros(batch_size, m - n, d_model).type(torch.cuda.FloatTensor)],
                                   1)
            tgt_seq = torch.cat([tgt_seq, torch.zeros(batch_size, m - n).type(torch.cuda.LongTensor)], 1)

        combined_dec_output = dec_output * gates + retrieval_dec_output * (1-gates)
        # combined_dec_output = dec_output + retrieval_dec_output

        non_pad_mask = get_non_pad_mask(tgt_seq)

        slf_attn_mask_subseq = get_subsequent_mask(tgt_seq).type(torch.cuda.LongTensor)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=tgt_seq, seq_q=tgt_seq).type(torch.cuda.LongTensor)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)

        dec_enc_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=tgt_seq)

        combined_dec_output, dec_slf_attn, dec_enc_attn = self.last_layer(combined_dec_output, enc_output,
            non_pad_mask=non_pad_mask,
            slf_attn_mask=slf_attn_mask,
            dec_enc_attn_mask=dec_enc_attn_mask)

        # if return_attns:
        #     return dec_output, dec_slf_attn_list, dec_enc_attn_list

        return combined_dec_output,


class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self,
            n_src_vocab, n_tgt_vocab, len_max_seq,
            d_word_vec=512, d_model=512, d_inner=2048,
            n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1,
            tgt_emb_prj_weight_sharing=True,
            emb_src_tgt_weight_sharing=True,
            embedding_matrix=None):

        super().__init__()

        # d_word_vec and d_model are different, but will be transformed later
        self.encoder = Encoder(
            n_src_vocab=n_src_vocab, len_max_seq=len_max_seq,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout, embedding_matrix=embedding_matrix)

        # use d_model as d_word_vec
        self.decoder = Decoder(
            n_tgt_vocab=n_tgt_vocab, len_max_seq=len_max_seq,
            d_word_vec=d_model, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout)

        print("set tgt embedding, size {}".format(self.decoder.tgt_word_emb.weight.size()))
        print("set positions decoder, size{}".format(self.decoder.position_enc.weight.size()))
        self.tgt_word_prj = nn.Linear(d_model, n_tgt_vocab, bias=False)
        # self.tgt_word_prj = nn.Linear(2 * d_model, n_tgt_vocab, bias=False)
        nn.init.xavier_normal_(self.tgt_word_prj.weight)

        # self.aux_tgt_word_prj = nn.Linear(d_model, n_tgt_vocab, bias=False)
        # nn.init.xavier_normal_(self.aux_tgt_word_prj.weight)


        if tgt_emb_prj_weight_sharing:
            # Share the weight matrix between target word embedding & the final logit dense layer
            self.tgt_word_prj.weight = self.decoder.tgt_word_emb.weight
            self.x_logit_scale = (d_model ** -0.5)
        else:
            self.x_logit_scale = 1.

        if emb_src_tgt_weight_sharing:
            # Share the weight matrix between source & target word embeddings
            assert n_src_vocab == n_tgt_vocab, \
            "To share word embedding table, the vocabulary size of src/tgt shall be the same."
            self.encoder.src_word_emb.weight = self.decoder.tgt_word_emb.weight

        # num_inputs, num_outputs, controller_size, controller_layers, num_heads, N, M, K, controller_type = 'MLP')
        self.gcl = EncapsulatedGCL(d_model, d_model, 128, 2, 1, N=64, M=128, K=d_model, controller_type='MLP')
        self.gcl.init_sequence(1)

    def find_match(self, k):
        _, loc = self.gcl.memory.address(k, 1, None, None, 1, flip_keys=False)
        match_k = self.gcl.memory.keys[loc]

        return match_k

    def forward(self, src_seq, src_pos, tgt_seq, tgt_pos):
        tgt_seq, tgt_pos = tgt_seq[:, :-1], tgt_pos[:, :-1]

        enc_output, *_ = self.encoder(src_seq, src_pos)
        dec_output, *_ = self.decoder(tgt_seq, tgt_pos, src_seq, enc_output)

        # k = enc_output.clone()
        # k = F.max_pool1d(k.permute(0, 2, 1), k.shape[-2]).squeeze()
        # x = dec_output.clone()
        # x = F.max_pool1d(x.permute(0, 2, 1), x.shape[-2]).squeeze()
        #
        # gcl_output = self.gcl(k.unsqueeze(0).detach(), x.unsqueeze(0), bidirectional=False, save_attn=False)
        # # print(enc_output.shape, gcl_output.shape)
        # time_steps = dec_output.shape[1]
        # gcl_output = gcl_output.permute(1, 0, 2).repeat(1, time_steps, 1)
        #
        # # dec_output = dec_output + gcl_output.squeeze()
        # dec_output = torch.cat([dec_output, gcl_output.squeeze()], -1)

        seq_logit = self.tgt_word_prj(dec_output) * self.x_logit_scale

        return seq_logit.view(-1, seq_logit.size(2))


def reverse_tensor(tensor, dim):
    reverse_idx = [i for i in range(tensor.size(dim)-1, -1, -1)]
    if torch.cuda.is_available():
        reverse_idx = torch.cuda.LongTensor(reverse_idx)
    else:
        reverse_idx = torch.LongTensor(reverse_idx)

    return tensor.index_select(dim, reverse_idx)


class BiTransformer(nn.Module):
    def __init__(
            self,
            n_src_vocab, n_tgt_vocab, len_max_seq,
            d_word_vec=512, d_model=512, d_inner=2048,
            n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1,
            tgt_emb_prj_weight_sharing=False,
            emb_src_tgt_weight_sharing=False,
            embedding_matrix=None):

        super().__init__()

        args = locals()
        del args['self']
        del args['__class__']
        args['tgt_emb_prj_weight_sharing'] = False
        args['emb_src_tgt_weight_sharing'] = False

        self.transformer = Transformer(**args)
        self.encoder = self.transformer.encoder
        self.decoder_lr = self.transformer.decoder  # left to right, same as original transformer decoder
        self.decoder_rl = copy.deepcopy(self.decoder_lr)  # the same architecture

        self.tgt_word_prj_lr = nn.Linear(d_model, n_tgt_vocab, bias=False)
        self.tgt_word_prj_rl = nn.Linear(d_model, n_tgt_vocab, bias=False)
        nn.init.xavier_normal_(self.tgt_word_prj_lr.weight)
        nn.init.xavier_normal_(self.tgt_word_prj_rl.weight)

        self.aux_tgt_word_prj_lr = nn.Linear(d_model, n_tgt_vocab, bias=False)
        self.aux_tgt_word_prj_rl = nn.Linear(d_model, n_tgt_vocab, bias=False)
        nn.init.xavier_normal_(self.aux_tgt_word_prj_lr.weight)
        nn.init.xavier_normal_(self.aux_tgt_word_prj_rl.weight)

    def forward(self, src_seq, src_pos, tgt_seq, tgt_seq_reversed, tgt_pos, return_intermediate=False):

        tgt_seq, tgt_seq_reversed, tgt_pos = tgt_seq[:, :-1], tgt_seq_reversed[:, :-1], tgt_pos[:, :-1]
        # print(tgt_seq_reversed)

        enc_output, *_ = self.encoder(src_seq, src_pos)

        dec_output_lr, *_ = self.decoder_lr(tgt_seq, tgt_pos, src_seq, enc_output)
        dec_output_rl, *_ = self.decoder_rl(tgt_seq_reversed, tgt_pos, src_seq, enc_output)

        seq_logit_lr = self.tgt_word_prj_lr(dec_output_lr)
        seq_logit_rl = self.tgt_word_prj_rl(dec_output_rl)

        return seq_logit_lr.view(-1, seq_logit_lr.size(2)), seq_logit_rl.view(-1, seq_logit_rl.size(2))


class Translator(OriginalTranslator):
    def __init__(self, opt, model=None):
        if not hasattr(opt, 'beam_size'):
            opt.beam_size = 5
        if not hasattr(opt, 'n_best'):
            opt.n_best = 1

        self.opt = opt
        self.device = torch.device('cuda' if opt.cuda else 'cpu')
        self.retrieve_only = False

        if model is None:
            checkpoint = torch.load(opt.model)
            model_opt = checkpoint['settings']
            self.model_opt = model_opt
            self.opt.bi = self.model_opt.bi
            print("Bidirectional = {}".format(self.opt.bi))

            if self.opt.bi:
                model = BiTransformer(
                    model_opt.src_vocab_size,
                    model_opt.tgt_vocab_size,
                    model_opt.max_token_seq_len,
                    tgt_emb_prj_weight_sharing=model_opt.proj_share_weight,
                    emb_src_tgt_weight_sharing=model_opt.embs_share_weight,
                    d_k=model_opt.d_k,
                    d_v=model_opt.d_v,
                    d_model=model_opt.d_model,
                    d_word_vec=model_opt.d_word_vec,  # src word vector dimension
                    d_inner=model_opt.d_inner_hid,
                    n_layers=model_opt.n_layers,
                    n_head=model_opt.n_head,
                    dropout=model_opt.dropout)
            else:
                model = Transformer(
                    model_opt.src_vocab_size,
                    model_opt.tgt_vocab_size,
                    model_opt.max_token_seq_len,
                    tgt_emb_prj_weight_sharing=model_opt.proj_share_weight,
                    emb_src_tgt_weight_sharing=model_opt.embs_share_weight,
                    d_k=model_opt.d_k,
                    d_v=model_opt.d_v,
                    d_model=model_opt.d_model,
                    d_word_vec=model_opt.d_word_vec,
                    d_inner=model_opt.d_inner_hid,
                    n_layers=model_opt.n_layers,
                    n_head=model_opt.n_head,
                    dropout=model_opt.dropout)

            model.load_state_dict(checkpoint['model'])
            # model = checkpoint['model']
            if 'memory_lr' in checkpoint:
                model.decoder_lr.gcl.memory = checkpoint['memory_lr']
                model.decoder_rl.gcl.memory = checkpoint['memory_rl']
                model.decoder_lr.gcl.gcl.memory = model.decoder_lr.gcl.memory
                model.decoder_rl.gcl.gcl.memory = model.decoder_rl.gcl.memory
                for head in model.decoder_lr.gcl.gcl.heads:
                    head.memory = model.decoder_lr.gcl.memory
                for head in model.decoder_rl.gcl.gcl.heads:
                    head.memory = model.decoder_rl.gcl.memory
                print('memory loaded!', model.decoder_lr.gcl.gcl.memory.content[0])
            else:
                print('no memory found. ckpt keys:', checkpoint.keys())
            print('[Info] Trained model state loaded.')

        model.word_prob_prj = nn.LogSoftmax(dim=1)

        model = model.to(self.device)

        self.model = model
        # self.model.eval()

    def translate_batch(self, raw_src_seq, raw_src_pos, block_list=[]):
        ''' Translation work in one batch '''

        def retrieve_batch_tgt(src_seq, src_pos):

            src_seq, src_pos = src_seq.to(self.device), src_pos.to(self.device)
            enc_output, *_ = self.model.encoder(src_seq, src_pos)

            pred_seq_lr, pos_lr = self.model.decoder_lr(None, None, src_seq, enc_output, retrieve_only=True)
            pred_seq_rl, pos_rl = self.model.decoder_rl(None, None, src_seq, enc_output, retrieve_only=True)

            pred_seq_lr = pred_seq_lr.tolist()
            pred_seq_rl = pred_seq_rl.tolist()

            batch_size = src_seq.shape[0]
            batch_hyp_list = []  # list of results from each decoder
            batch_scores_list = []

            for i in range(batch_size):
                batch_hyp_list.append([[[x for x in pred_seq_lr[i] if x not in (0, 1, 2, 3)]],
                                       [[x for x in pred_seq_rl[i] if x not in (0, 1, 2, 3)]]]) # one best result for each decoder
                batch_scores_list.append([torch.ones(1).to(self.device), torch.ones(1).to(self.device)]) # dummy scores

            return batch_hyp_list, batch_scores_list


        def get_inst_idx_to_tensor_position_map(inst_idx_list):
            ''' Indicate the position of an instance in a tensor. '''
            return {inst_idx: tensor_position for tensor_position, inst_idx in enumerate(inst_idx_list)}

        def collect_active_part(beamed_tensor, curr_active_inst_idx, n_prev_active_inst, n_bm):
            ''' Collect tensor parts associated to active instances. '''

            _, *d_hs = beamed_tensor.size()
            n_curr_active_inst = len(curr_active_inst_idx)
            new_shape = (n_curr_active_inst * n_bm, *d_hs)

            beamed_tensor = beamed_tensor.view(n_prev_active_inst, -1)
            beamed_tensor = beamed_tensor.index_select(0, curr_active_inst_idx)
            beamed_tensor = beamed_tensor.view(*new_shape)

            return beamed_tensor

        def collate_active_info(
                src_seq, src_enc, inst_idx_to_position_map, active_inst_idx_list):
            # Sentences which are still active are collected,
            # so the decoder will not run on completed sentences.
            n_prev_active_inst = len(inst_idx_to_position_map)
            active_inst_idx = [inst_idx_to_position_map[k] for k in active_inst_idx_list]
            active_inst_idx = torch.LongTensor(active_inst_idx).to(self.device)

            active_src_seq = collect_active_part(src_seq, active_inst_idx, n_prev_active_inst, n_bm)
            active_src_enc = collect_active_part(src_enc, active_inst_idx, n_prev_active_inst, n_bm)
            # print(active_src_enc.shape)
            if hasattr(self.model.encoder, 'ntm'):
                self.model.encoder.ntm.previous_state = list(self.model.encoder.ntm.previous_state)
                self.model.encoder.ntm.previous_state[1] = list(self.model.encoder.ntm.previous_state[1])
                memory = self.model.encoder.ntm.memory
                self.model.encoder.ntm.memory.memory = collect_active_part(memory.memory.view(n_prev_active_inst*n_bm, -1),
                                                                           active_inst_idx, n_prev_active_inst, n_bm).view(-1, memory.N, memory.M)
                self.model.encoder.ntm.memory.batch_size = self.model.encoder.ntm.memory.memory.shape[0]
                # print(self.model.encoder.ntm.memory.memory.shape, self.model.encoder.ntm.memory.batch_size)
                for i in range(len(self.model.encoder.ntm.previous_state)):
                    for j, tensor  in enumerate(self.model.encoder.ntm.previous_state[i]):
                        # print(i, j, tensor.shape)
                        squeezed = False
                        if len(tensor.shape) == 3:
                            dim0, dim1, dim2 = tensor.shape  # dim1 = n_prev_active_inst*n_bm
                            tensor = torch.transpose(tensor, 0, 1).contiguous().view(dim1, -1)
                            # tensor = tensor.squeeze(0)
                            squeezed = True
                        new_tensor = collect_active_part(tensor, active_inst_idx, n_prev_active_inst, n_bm)
                        if squeezed:
                            new_tensor = torch.transpose(new_tensor.contiguous().view(-1, dim0, dim2), 0, 1).contiguous()

                        # print(new_tensor.shape)
                        self.model.encoder.ntm.previous_state[i][j] = new_tensor

            # active_src_enc.register_hook(print_grad('active src enc'))
            active_src_enc[torch.isnan(active_src_enc)] = 0
            active_inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)

            return active_src_seq, active_src_enc, active_inst_idx_to_position_map

        def prepare_beam_dec_seq(inst_dec_beams, len_dec_seq):
            dec_partial_seq = [b.get_current_state() for b in inst_dec_beams if not b.done]
            dec_partial_seq = torch.stack(dec_partial_seq).to(self.device)
            dec_partial_seq = dec_partial_seq.view(-1, len_dec_seq)
            return dec_partial_seq

        def prepare_beam_dec_pos(len_dec_seq, n_active_inst, n_bm):
            dec_partial_pos = torch.arange(1, len_dec_seq + 1, dtype=torch.long, device=self.device)
            dec_partial_pos = dec_partial_pos.unsqueeze(0).repeat(n_active_inst * n_bm, 1)
            return dec_partial_pos

        def predict_word(decoder, tgt_word_prj, dec_seq, dec_pos, src_seq, enc_output, n_active_inst, n_bm):
            """decoder is added as an argument, compared to the original version"""
            # sometimes the output is only [0] the pad token

            dec_output, *_ = decoder(dec_seq, dec_pos, src_seq, enc_output)
            dec_output_last = dec_output[:, -1, :]  # Pick the last step: (bh * bm) * d_h
            # dec_output[torch.isnan(dec_output)] = 0

            # dec_output.register_hook(print_grad('dec_output in predict_word {} data'.format(dec_seq)))
            # print(dec_output)

            # gcl decoder
            # k = enc_output.clone()
            # k = F.max_pool1d(k.permute(0, 2, 1), k.shape[-2]).squeeze()
            # x = dec_output.clone()
            # x = F.max_pool1d(x.permute(0, 2, 1), x.shape[-2]).squeeze()
            #
            # gcl_output = self.model.gcl(k.unsqueeze(0).detach(), x.unsqueeze(0), bidirectional=False, save_attn=False)
            #
            # dec_output_last = torch.cat([dec_output_last, gcl_output.squeeze()], -1)

            word_prob = F.log_softmax(tgt_word_prj(dec_output_last), dim=1)
            # word_prob.register_hook(print_grad('word prob in predict_word'))
            word_prob = word_prob.view(n_active_inst, n_bm, -1)
            if block_list != []:
                for block_tok in block_list:
                    word_prob[:,:,block_tok] = -1000.

            return word_prob

        def collect_active_inst_idx_list(inst_beams, word_prob, inst_idx_to_position_map):
            """get indexes of instances that have not been fully translated yet"""
            active_inst_idx_list = []
            for inst_idx, inst_position in inst_idx_to_position_map.items():
                is_inst_complete = inst_beams[inst_idx].advance(word_prob[inst_position])
                if not is_inst_complete:
                    active_inst_idx_list += [inst_idx]

            return active_inst_idx_list

        def beam_decode_step(
                decoder, tgt_word_prj, inst_dec_beams, len_dec_seq, src_seq, enc_output, inst_idx_to_position_map, n_bm):
            ''' Decode and update beam status, and then return active beam idx
                decoder is added as an argument, compared to the original version
            '''

            # enc_output.register_hook(print_grad('enc output'))
            n_active_inst = len(inst_idx_to_position_map)

            dec_seq = prepare_beam_dec_seq(inst_dec_beams, len_dec_seq)
            dec_pos = prepare_beam_dec_pos(len_dec_seq, n_active_inst, n_bm)
            word_prob = predict_word(decoder, tgt_word_prj, dec_seq, dec_pos, src_seq, enc_output, n_active_inst, n_bm)
            # word_prob.register_hook(print_grad('word prob in beam decode'))  # grad ok

            # Update the beam with predicted word prob information and collect incomplete instances
            active_inst_idx_list = collect_active_inst_idx_list(
                inst_dec_beams, word_prob, inst_idx_to_position_map)

            return active_inst_idx_list

        def collect_hypothesis_and_scores(inst_dec_beams, n_best):
            all_hyp, all_scores = [], []
            for inst_idx in range(len(inst_dec_beams)):
                scores, tail_idxs = inst_dec_beams[inst_idx].sort_scores()
                # scores.register_hook(print_grad('scores from collect hypothses'))
                all_scores = all_scores + [scores[:n_best]]

                hyps = [inst_dec_beams[inst_idx].get_hypothesis(i) for i in tail_idxs[:n_best]]
                all_hyp += [hyps]
            return all_hyp, all_scores


        if self.retrieve_only:
            batch_hyp_list, batch_scores_list = retrieve_batch_tgt(raw_src_seq, raw_src_pos)
            return batch_hyp_list, batch_scores_list

        if self.opt.bi:
            decoders = [self.model.decoder_lr, self.model.decoder_rl]
            tgt_word_prjs = [self.model.tgt_word_prj_lr, self.model.tgt_word_prj_rl]
        else:
            decoders = [self.model.decoder]
            tgt_word_prjs = [self.model.tgt_word_prj]

        batch_hyp_list = []  # list of results from each decoder
        batch_scores_list = []

        n_bm = self.opt.beam_size

        # -- Decode
        for decoder, tgt_word_prj in zip(decoders, tgt_word_prjs):  # two decoders for bidirectional model
            src_seq = copy.copy(raw_src_seq)
            src_pos = copy.copy(raw_src_pos)

            # -- Encode
            src_seq, src_pos = src_seq.to(self.device), src_pos.to(self.device)
            if hasattr(self.model.encoder, 'ntm'):
                n_inst, len_s = src_seq.size()
                src_seq = src_seq.repeat(1, n_bm).view(n_inst * n_bm, len_s)
            src_enc, *_ = self.model.encoder(src_seq, src_pos)

            #-- Repeat data for beam search
            if len(src_enc.size()) == 3:
                n_inst, len_s, d_h = src_enc.size()
                src_seq = src_seq.repeat(1, n_bm).view(n_inst * n_bm, len_s)
                src_enc = src_enc.unsqueeze(1).expand(-1, n_bm, -1, -1).contiguous().view(n_inst * n_bm, len_s, d_h)

            # src_enc.register_hook(print_grad('{}, src_enc'.format(src_enc.size())))

            #-- Prepare beams
            inst_dec_beams = [Beam(n_bm, device=self.device) for _ in range(n_inst)]

            # -- Bookkeeping for active or not
            active_inst_idx_list = list(range(n_inst))
            inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)

            for len_dec_seq in range(1, self.model_opt.max_token_seq_len + 1):

                # if len_dec_seq > 30:  # abnormally long seq
                #     print(len_dec_seq)

                active_inst_idx_list = beam_decode_step(
                    decoder, tgt_word_prj, inst_dec_beams, len_dec_seq, src_seq, src_enc, inst_idx_to_position_map, n_bm)

                # src_enc[torch.isnan(src_enc)] = 0
                if not active_inst_idx_list:
                    break  # all instances have finished their path to <EOS>

                src_seq, src_enc, inst_idx_to_position_map = collate_active_info(
                    src_seq, src_enc, inst_idx_to_position_map, active_inst_idx_list)
                # src_enc.register_hook(print_grad('active src enc'))  # GRAD OK HERE

            # batch_hyp is a nested list of [batches [n_best seqs] ]
            batch_hyp, batch_scores = collect_hypothesis_and_scores(inst_dec_beams, self.opt.n_best)
            # print('\n')
            # print(batch_scores)
            batch_hyp_list.append(batch_hyp)
            batch_scores_list.append(batch_scores)
        # print(batch_hyp_list, batch_scores_list);sys.exit(1)

        return batch_hyp_list, batch_scores_list
