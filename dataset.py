import numpy as np
import torch
import torch.utils.data
import config
from transformer import Constants
import sys
from src.math_laws import permute_tgt

#
# def paired_collate_fn(insts):
#     src_insts, tgt_insts, tgt_nums_insts = list(zip(*insts))
#     src_insts = collate_fn(src_insts)
#     tgt_insts = collate_fn(tgt_insts)
#     tgt_nums_insts = collate_fn(tgt_nums_insts, max_len=tgt_insts[0].size()[-1])
#     return (*src_insts, *tgt_insts, *tgt_nums_insts)
#
# def bidirectional_collate_fn(insts):
#     src_insts, tgt_insts, tgt_nums_insts = list(zip(*insts))
#     src_insts = collate_fn(src_insts)
#     tgt_insts = collate_fn(tgt_insts, reverse=True)
#     tgt_nums_insts = collate_fn(tgt_nums_insts, reverse=True, max_len=tgt_insts[0].size()[-1])
#     return (*src_insts, *tgt_insts, *tgt_nums_insts)
#
#
# def collate_fn(insts, reverse=False, max_len=None):
#     ''' Pad the instance to the max seq length in batch '''
#
#     if max_len is None:
#         max_len = max(len(inst) for inst in insts)
#
#     batch_seq = np.array([
#             inst + [Constants.PAD] * (max_len - len(inst))
#             for inst in insts])
#
#     if reverse:
#         # replace BOS with EOS
#         batch_seq_reversed = np.array([
#             [x if x!=Constants.BOS else Constants.EOS for x in inst[::-1]] + [Constants.PAD] * (max_len - len(inst))
#             for inst in insts])
#         batch_seq_reversed[:, 0] = Constants.BOS
#
#     batch_pos = np.array([
#         [pos_i+1 if w_i != Constants.PAD else 0
#          for pos_i, w_i in enumerate(inst)] for inst in batch_seq])
#
#     batch_seq = torch.LongTensor(batch_seq)
#     batch_pos = torch.LongTensor(batch_pos)
#
#     if reverse:
#         batch_seq_reversed = torch.LongTensor(batch_seq_reversed)
#         return batch_seq, batch_seq_reversed, batch_pos
#     return batch_seq, batch_pos

class TranslationDataset(torch.utils.data.Dataset):
    def __init__(
        self, src_word2idx, tgt_word2idx,
        src_insts=None, tgt_insts=None, tgt_nums=None, permute_tgt=False):

        assert src_insts
        assert not tgt_insts or (len(src_insts) == len(tgt_insts))

        src_idx2word = {idx:word for word, idx in src_word2idx.items()}
        self._src_word2idx = src_word2idx
        self._src_idx2word = src_idx2word
        self._src_insts = src_insts

        tgt_idx2word = {idx:word for word, idx in tgt_word2idx.items()}
        self._tgt_word2idx = tgt_word2idx
        self._tgt_idx2word = tgt_idx2word
        self._tgt_insts = tgt_insts

        self._tgt_nums = tgt_nums
        self.permute_tgt = permute_tgt

    @property
    def n_insts(self):
        ''' Property for dataset size '''
        return len(self._src_insts)

    @property
    def src_vocab_size(self):
        ''' Property for vocab size '''
        return len(self._src_word2idx)

    @property
    def tgt_vocab_size(self):
        ''' Property for vocab size '''
        return len(self._tgt_word2idx)

    @property
    def src_word2idx(self):
        ''' Property for word dictionary '''
        return self._src_word2idx

    @property
    def tgt_word2idx(self):
        ''' Property for word dictionary '''
        return self._tgt_word2idx

    @property
    def src_idx2word(self):
        ''' Property for index dictionary '''
        return self._src_idx2word

    @property
    def tgt_idx2word(self):
        ''' Property for index dictionary '''
        return self._tgt_idx2word

    def __len__(self):
        return self.n_insts

    def __getitem__(self, idx):
        if self._tgt_insts:
            return self._src_insts[idx], self._tgt_insts[idx], self._tgt_nums[idx]
        return self._src_insts[idx]

    def paired_collate_fn(self, insts):
        src_insts, tgt_insts, tgt_nums_insts = list(zip(*insts))
        src_insts = self.collate_fn(src_insts)
        tgt_insts = self.collate_fn(tgt_insts)
        tgt_nums_insts = self.collate_fn(tgt_nums_insts, max_len=tgt_insts[0].size()[-1])
        return (*src_insts, *tgt_insts, *tgt_nums_insts)

    def bidirectional_collate_fn(self, insts):
        src_insts, tgt_insts, tgt_nums_insts = list(zip(*insts))
        if self.permute_tgt:
            tgt_insts, tgt_nums_insts = permute_tgt(tgt_insts, tgt_tok2idx_dict=self._tgt_word2idx)  # permute equations

        src_insts = self.collate_fn(src_insts)
        tgt_insts = self.collate_fn(tgt_insts, reverse=True)
        tgt_nums_insts = self.collate_fn(tgt_nums_insts, reverse=True, max_len=tgt_insts[0].size()[-1])
        return (*src_insts, *tgt_insts, *tgt_nums_insts)

    def collate_fn(self, insts, reverse=False, max_len=None):
        ''' Pad the instance to the max seq length in batch '''

        if max_len is None:
            max_len = max(len(inst) for inst in insts)

        batch_seq = np.array([
            inst + [Constants.PAD] * (max_len - len(inst))
            for inst in insts])

        if reverse:
            # replace BOS with EOS
            batch_seq_reversed = np.array([
                [x if x != Constants.BOS else Constants.EOS for x in inst[::-1]] + [Constants.PAD] * (
                            max_len - len(inst))
                for inst in insts])
            batch_seq_reversed[:, 0] = Constants.BOS

        batch_pos = np.array([
            [pos_i + 1 if w_i != Constants.PAD else 0
             for pos_i, w_i in enumerate(inst)] for inst in batch_seq])

        batch_seq = torch.LongTensor(batch_seq)
        batch_pos = torch.LongTensor(batch_pos)

        if reverse:
            batch_seq_reversed = torch.LongTensor(batch_seq_reversed)
            return batch_seq, batch_seq_reversed, batch_pos

        return batch_seq, batch_pos
