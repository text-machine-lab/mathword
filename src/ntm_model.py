import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
import transformer.Constants as Constants
from src.model import Translator
import random
import numpy as np

from ntm.aio import EncapsulatedNTM


class Seq2SeqMem(nn.Module):
    """memory enhanced model"""

    def __init__(self, src_vocab_size, tgt_vocab_size, d_word_vec, d_tgt_vec, d_ntm_input, d_controller, d_sent_enc, n_layers, n_heads, n_slots, m_depth, d_dec_output):
        super().__init__()
        self.encoder = NTMEncoder(src_vocab_size, d_word_vec, d_ntm_input, d_controller, d_sent_enc, n_layers, n_heads, n_slots, m_depth)
        self.decoder = NTMDecoder(self.encoder.ntm, d_ntm_input, tgt_vocab_size, d_tgt_vec, d_dec_output, d_sent_enc)
        self.tgt_word_prj = nn.Linear(d_dec_output, tgt_vocab_size)


    def forward(self, src_seq, src_pos, tgt_seq, tgt_pos):
        # position embeddings are not used. We keep them here only to match the format of transformer model,
        # so some modules for training/evaluation can be shared.
        tgt_seq, tgt_pos = tgt_seq[:, :-1], tgt_pos[:, :-1]
        # print("tgt_seq", tgt_seq.shape)

        enc_output, *_ = self.encoder(src_seq, src_pos)
        dec_output, *_ = self.decoder(tgt_seq, tgt_pos, src_seq, enc_output)
        # print(enc_output.shape, dec_output.shape)
        seq_logit = self.tgt_word_prj(torch.relu(dec_output))
        # print(seq_logit.shape)

        return seq_logit.view(-1, seq_logit.size(2))


class NTMEncoder(nn.Module):
    def __init__(self, src_vocab_size, d_word_vec, d_ntm_input, d_controller, d_sent_enc, n_layers, n_heads, n_slots, m_depth, embedding_matrix=None):
        super().__init__()
        if embedding_matrix is None:
            self.src_word_emb = nn.Embedding(src_vocab_size, d_word_vec, padding_idx=Constants.PAD)
        else:
            self.src_word_emb = nn.Embedding.from_pretrained(
                torch.from_numpy(embedding_matrix).type(torch.cuda.FloatTensor),freeze=True)

            print("set pretrained word embeddings, size {}".format(self.src_word_emb.weight.size()))

        self.linear = nn.Linear(d_word_vec, d_ntm_input)
        self.ntm = EncapsulatedNTM(d_ntm_input, d_sent_enc, d_controller, n_layers, n_heads, n_slots, m_depth)

    def forward(self, src_seq, src_pos):
        """
        :param src_seq: source token index list
        :param src_pos: not used.
        :return:
        """
        batch_size, seq_len = src_seq.shape
        self.ntm.init_sequence(batch_size)

        embs = self.src_word_emb(src_seq)
        ntm_input = self.linear(F.dropout(embs, p=0.3))

        for t in range(seq_len):
            output, state = self.ntm(ntm_input[:,t,:])

        return output,


class NTMDecoder(nn.Module):
    def __init__(self, ntm, d_ntm_input, tgt_vocab_size, d_tgt_vec, d_dec_output, d_sent_enc):
        super().__init__()
        self.embs = nn.Embedding(tgt_vocab_size, d_tgt_vec)
        self.linear = nn.Linear(d_tgt_vec, d_ntm_input)  # hidden layer between input and ntm
        self.ntm = ntm
        self.output = nn.Linear(d_sent_enc, d_dec_output)  # decoder output layer


    def forward(self, tgt_seq, tgt_pos, src_seq, enc_output, sample_func=None):
        # tgt_seq, tgt_pos, src_seq, enc_output,

        # if sample_func is None:
        #     sample_func = partial(torch.argmax, dim=-1)

        # state = self.init.expand(b, -1)  # repeat across batch dimension
        batch_size, seq_len = tgt_seq.shape
        tgt_emb = self.embs(tgt_seq)
        dec_input = self.linear(tgt_emb)

        all_dec_output = []
        for t in range(seq_len):
            ntm_out, _ = self.ntm(dec_input[:,t,:])  # output has sigmoid
            ntm_out = F.dropout(torch.relu(ntm_out), p=0.2)
            dec_output = self.output(ntm_out)
            all_dec_output.append(dec_output)

        return torch.stack(all_dec_output, dim=1),


class NTMTranslator(Translator):
    def __init__(self, opt, model=None):
        if not hasattr(opt, 'beam_size'):
            opt.beam_size = 5
        if not hasattr(opt, 'n_best'):
            opt.n_best = 1

        self.opt = opt
        self.device = torch.device('cuda' if opt.cuda else 'cpu')

        if model is None:
            checkpoint = torch.load(opt.model)
            model_opt = checkpoint['settings']
            self.model_opt = model_opt
            self.opt.bi = self.model_opt.bi

            model = Seq2SeqMem(model_opt.src_vocab_size,
                               model_opt.tgt_vocab_size,
                               model_opt.d_word_vec,
                               model_opt.d_tgt_vec,
                               model_opt.d_ntm_input,
                               model_opt.d_controller,
                               model_opt.d_sent_enc,
                               model_opt.n_controller_layers,
                               model_opt.n_heads,
                               model_opt.n_slots,
                               model_opt.m_depth,
                               model_opt.d_dec_output)

            model.load_state_dict(checkpoint['model'])
            print('[Info] Trained model state loaded.')

        model.word_prob_prj = nn.LogSoftmax(dim=1)

        model = model.to(self.device)

        self.model = model