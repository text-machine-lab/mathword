import torch
import torch.nn as nn
import transformer.Constants as Constants
from transformer.Models import Encoder as OriginalEncoder
from transformer.Models import get_sinusoid_encoding_table, EncoderLayer, Decoder, get_non_pad_mask, get_attn_key_pad_mask
from transformer.Translator import Translator as OriginalTranslator


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
                torch.from_numpy(embedding_matrix).type(torch.cuda.FloatTensor),
                freeze = False
            )
            print("set pretrained word embeddings, size {}".format(self.src_word_emb.weight.size()))

        self.src_word_enc = nn.Linear(d_word_vec, d_model)  # need this because d_word_vec != d_model

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
        enc_output = self.src_word_enc(self.src_word_emb(src_seq)) + self.position_enc(src_pos)

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

        self.tgt_word_prj = nn.Linear(d_model, n_tgt_vocab, bias=False)
        nn.init.xavier_normal_(self.tgt_word_prj.weight)

        # assert d_model == d_word_vec, \
        # 'To facilitate the residual connections, \
        #  the dimensions of all module outputs shall be the same.'

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

    def forward(self, src_seq, src_pos, tgt_seq, tgt_pos):

        tgt_seq, tgt_pos = tgt_seq[:, :-1], tgt_pos[:, :-1]

        enc_output, *_ = self.encoder(src_seq, src_pos)
        dec_output, *_ = self.decoder(tgt_seq, tgt_pos, src_seq, enc_output)
        seq_logit = self.tgt_word_prj(dec_output) * self.x_logit_scale

        return seq_logit.view(-1, seq_logit.size(2))


class Translator(OriginalTranslator):
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
            print('[Info] Trained model state loaded.')

        model.word_prob_prj = nn.LogSoftmax(dim=1)

        model = model.to(self.device)

        self.model = model
        self.model.eval()