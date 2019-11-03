import argparse
import torch
import torch.optim as optim
import config
from src.ntm_model import Seq2SeqMem
from train import Scheduler, prepare_dataloaders
from train import train_transformer as train_model


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-data', required=True)
    parser.add_argument('-split', type=float, default=0.8, help="portion for training")

    parser.add_argument('-epoch', type=int, default=100)
    parser.add_argument('-batch_size', type=int, default=16)

    parser.add_argument('-d_word_vec', type=int, default=300,
                        help="dimension of src text word vectors")
    parser.add_argument('-d_tgt_vec', type=int, default=300,
                        help="dimension of target equation symbol vectors")
    parser.add_argument('-d_ntm_input', type=int, default=512,
                        help="dimension of ntm input")
    parser.add_argument('-d_controller', type=int, default=1024,
                        help='size of ntm controller')
    parser.add_argument('-d_sent_enc', type=int, default=512,
                        help='size of sentence encoder, i.e. ntm output size')
    parser.add_argument('-d_dec_output', type=int, default=256,
                        help='dimentino of decoder output i.e. hidden layer size before output layer')

    parser.add_argument('-n_heads', type=int, default=1)
    parser.add_argument('-n_slots', type=int, default=64, help="number of memory slots")
    parser.add_argument('-m_depth', type=int, default=512, help="memory capacity (depth)")
    parser.add_argument('-n_controller_layers', type=int, default=2)
    # parser.add_argument('-n_warmup_steps', type=int, default=4000)

    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-log', default=None)
    parser.add_argument('-save_model', default=None)
    parser.add_argument('-save_mode', type=str, choices=['all', 'best', 'last'], default='last')

    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-label_smoothing', action='store_true')
    parser.add_argument('-load_model', type=str, default=None, help='load pretrained model')

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda
    opt.bi = False  # this is for transformer model
    # opt.d_word_vec = opt.d_model

    # ========= Loading Dataset =========#
    data = torch.load(opt.data)
    opt.max_token_seq_len = data['settings']['max_token_seq_len']

    training_data, validation_data = prepare_dataloaders(data, opt)

    opt.src_vocab_size = training_data.dataset.src_vocab_size
    opt.tgt_vocab_size = training_data.dataset.tgt_vocab_size

    ops_idx = [data['dict']['tgt'][s] for s in ('+', '-', '*', '/')]
    opt.ops_idx = ops_idx  # indexes of operators
    # opt.ops_idx = None


    # ========= Preparing Model =========#
    # if opt.embs_share_weight:
    #     assert training_data.dataset.src_word2idx == training_data.dataset.tgt_word2idx, \
    #         'The src/tgt word2idx table are different but asked to share word embedding.'

    print(opt)

    device = torch.device('cuda' if opt.cuda else 'cpu')
    model = Seq2SeqMem(opt.src_vocab_size,
                       opt.tgt_vocab_size,
                       opt.d_word_vec,
                       opt.d_tgt_vec,
                       opt.d_ntm_input,
                       opt.d_controller,
                       opt.d_sent_enc,
                       opt.n_controller_layers,
                       opt.n_heads,
                       opt.n_slots,
                       opt.m_depth,
                       opt.d_dec_output).to(device)

    optimizer = Scheduler(
        optim.Adam(
            filter(lambda x: x.requires_grad, model.parameters()),
            betas=(0.9, 0.98), eps=1e-09),
            alpha=1e-4, n_current_steps=0)  # 5e-6

    if opt.load_model is not None:
        checkpoint = torch.load(opt.load_model)
        model.load_state_dict(checkpoint['model'])

    train_model(model, training_data, validation_data, optimizer, device, opt)


if __name__ == '__main__':
    main()