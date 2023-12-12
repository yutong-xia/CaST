import torch
import numpy as np
import os
import time
import argparse
import yaml
import pickle
import scipy.sparse as sp
from scipy.sparse import linalg

import torch.nn as nn
import torch

from src.utils.helper import get_dataloader_cast, check_device, get_num_nodes, get_num_edges, setup_seed, get_null_value
from src.utils.metrics import masked_mae
from src.models.cast import CaST
from src.trainers.cast_trainer import CaST_Trainer
from src.utils.args import get_public_config, str_to_bool
from src.utils.graph_algo import load_graph_data

def get_config():
    parser = get_public_config()

    # get private config
    parser.add_argument('--model_name', type=str, default='cast', help='model name')
    parser.add_argument('--hid_dim', type=int, default=16)
    parser.add_argument('--node_embed_dim', type=int, default=5)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--filter_type', type=str, default=None)
    parser.add_argument('--seed', type=int, default=0)

    # loss
    parser.add_argument('--beta1', type=float, default=1.0, help='contribution of commitment loss')
    parser.add_argument('--beta2', type=float, default=1.0, help='contribution of mutual information regulization loss')

    # temporal
    parser.add_argument('--depth', type=int, default=10, help='hp for temporal block')
    parser.add_argument('--n_envs', type=int, default=5, help='the number of environments')
    
    # spatial
    parser.add_argument('--K', type=int, default=2, help='num of polynomial')
    parser.add_argument('--bias', type=str_to_bool, default=True, help='whether to use bias')

    # training
    parser.add_argument('--base_lr', type=float, default=0.001)
    parser.add_argument('--lr_decay_ratio', type=float, default=0.7)
    
    # dataset
    parser.add_argument('--interval', type=int, default=1)
    parser.add_argument('--train_ratio',type=float ,default=8/12, help='The training set ratio')
    parser.add_argument('--val_ratio',type=float ,default=2/12, help='The validation set ratio')
    parser.add_argument('--train_val',type=str ,default='8_2_2', help='just for the name of the results')
    parser.add_argument('--time_delay_scaler', type=int, default=6, help='the rolling step on time series when calculating the time delay DTW')
    
    args = parser.parse_args()
    args.steps = [10, 30, 50, 70, 80]
    
    print(args)
    
    args.folder_name = 'hid{}_dropout{}_lr{}_K{}_env{}_b1{}_b2{}_seed{}'.format(
                                                                args.hid_dim, 
                                                                args.dropout, 
                                                                args.base_lr,
                                                                args.K,
                                                                args.n_envs,
                                                                args.beta1,
                                                                args.beta2,
                                                                args.seed)
    args.log_dir = './logs/{}/{}_{}_{}_{}/{}/{}/'.format(args.dataset+ '_' + args.train_val,
                                                        args.seq_len, args.horizon, args.input_dim, args.output_dim,
                                                        args.model_name,
                                                        args.folder_name)
    args.num_nodes = get_num_nodes(args.dataset)  
    args.num_edges = get_num_edges(args.dataset)  
    args.null_value = get_null_value(args.dataset)
                                       
    if args.filter_type == 'identity':
        args.support_len = 1
    else:
        args.support_len = 3

    args.datapath = os.path.join('./data', args.dataset)
    if args.dataset[:3] == 'AIR':
        args.graph_pkl = 'data/sensor_graph/adj_mx_{}.pkl'.format(args.dataset.lower()[:6])
    else:
        args.graph_pkl = 'data/sensor_graph/adj_mx_{}.pkl'.format(args.dataset.lower())
    if args.seed != 0:
        setup_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    return args

def main():
    args = get_config()
    device = check_device()
    _, _, adj_mat = load_graph_data(args.graph_pkl)
    

    model = CaST(name = args.model_name,
                dataset = args.dataset,
                device = device,
                num_nodes = args.num_nodes,
                num_edges = args.num_edges,
                seq_len = args.seq_len,
                horizon=args.horizon,
                input_dim=args.input_dim,
                output_dim=args.output_dim, 
                dropout=args.dropout,
                hid_dim = args.hid_dim,
                node_embed_dim = args.node_embed_dim,
                K = args.K, 
                depth = args.depth,
                bias = args.bias,
                time_delay_scaler = args.time_delay_scaler,
                n_envs = args.n_envs,
                )

    dataloader = get_dataloader_cast(datapath = args.datapath,
                          batch_size = args.batch_size,
                          input_dim = args.input_dim,
                          output_dim = args.output_dim,
                          seq_length_x = args.seq_len,
                          seq_length_y = args.horizon,
                          interval = args.interval,
                          time_delay_scaler = args.time_delay_scaler,
                          train_ratio = args.train_ratio,
                          val_ratio = args.val_ratio,
                          dataset_name=args.dataset
                          )

    result_path = args.result_path + '/' + args.dataset + '_' + args.train_val + '/{}_{}_{}_{}'.format(args.seq_len, args.horizon, args.input_dim, args.output_dim)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    trainer = CaST_Trainer(model=model,
                            adj_mat=adj_mat,
                            filter_type=args.filter_type,
                            data=dataloader,
                            base_lr=args.base_lr,
                            lr_decay_ratio=args.lr_decay_ratio,
                            log_dir=args.log_dir,
                            n_exp=args.n_exp,
                            save_iter=args.save_iter,
                            clip_grad_value=args.max_grad_norm,
                            max_epochs=args.max_epochs,
                            patience=args.patience,
                            device=device,
                            aug=args.aug,
                            steps=args.steps,
                            model_name = args.model_name,
                            result_path = result_path,
                            hp = args.folder_name,
                            beta1 = args.beta1,
                            beta2 = args.beta2,
                            null_value =args.null_value, 
                            )

    if args.mode == 'train':
        trainer.train()
        trainer.test(-1, 'test')
    else:
        trainer.test(-1, args.mode)
        if args.save_preds:
            trainer.save_preds(-1)


if __name__ == "__main__":
    main()