import os
import time
import argparse
import pickle

def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')

def get_public_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, help='train or test')
    parser.add_argument('--n_exp', type=int, default=None,
                        help='experiment index')
    parser.add_argument('--gpu', type=int, default=6, help='which gpu to run')

    # data
    parser.add_argument('--dataset', type=str, default='air_bj_new')
    parser.add_argument('--batch_size', type=int, default=1028)
    parser.add_argument('--aug', type=float, default=1.0)
    parser.add_argument('--seq_len', type=int, default=24)
    parser.add_argument('--horizon', type=int, default=24)
    parser.add_argument('--input_dim', type=int, default=1)
    parser.add_argument('--output_dim', type=int, default=1)

    # training
    parser.add_argument('--max_epochs', type=int, default=250) 
    parser.add_argument('--save_iter', type=int, default=400)
    parser.add_argument('--max_grad_norm', type=float, default=5.0)
    parser.add_argument('--patience', type=int, default=10)

    # test
    parser.add_argument('--save_preds', type=bool, default=False)
    parser.add_argument('--result_path', type=str, default='./results')
    return parser


