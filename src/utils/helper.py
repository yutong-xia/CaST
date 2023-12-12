from torch.utils.data import DataLoader
from src.utils.dataset import Dataset_CaST, Dataset_CaST_processed
import torch
from torch import Tensor
import logging
import numpy as np
import pandas as pd
import os
import sys
import pickle
import random

import torch_geometric

from src.utils.scaler import StandardScaler

def get_dataloader_cast(datapath, batch_size, input_dim, output_dim, seq_length_x, seq_length_y, interval, time_delay_scaler, train_ratio, val_ratio, dataset_name):
    processed = {}
    results = {}
    
    #### scaler
    scaler_dir = os.path.join(datapath, 'scaler.pkl')
    if not os.path.exists(scaler_dir):
        scalers = []
        data = np.load(os.path.join(datapath, 'dataset.npy'))
        for i in range(output_dim):
            scalers.append(StandardScaler(mean=data[..., i].mean(),
                                        std=data[..., i].std()))
        with open(scaler_dir, 'wb') as f:
            pickle.dump(scalers, f)
    else:
        with open(scaler_dir, 'rb') as f:
            scalers = pickle.load(f)
    
    
    #### dataset
    processed_dir = os.path.join(datapath, 'processed_{}_{}'.format(seq_length_x,seq_length_y))
    if not os.path.exists(processed_dir):
        os.mkdir(processed_dir)
        dataFile = os.path.join(datapath, 'dataset.npy')
        processed_dataset = Dataset_CaST(dataFile, datapath, scalers, input_dim, output_dim, seq_length_x, seq_length_y, interval, time_delay_scaler)
        dataloader = torch_geometric.loader.DataLoader(processed_dataset, batch_size = 1, shuffle=False)
        
        for i, data in enumerate(dataloader):
            [graph,  y] = data
            data_zip = {'graph':graph,
                        'y': y}
            torch.save(data_zip, os.path.join(processed_dir, 'Graph'+str(i)+'.pt'))
    
    num_samples = len(os.listdir(processed_dir)) - 3 # minus three label.csv
    print('num_samples: {}'.format(num_samples))

    idx_train = round(num_samples * train_ratio)
    idx_val = round(num_samples * (val_ratio+train_ratio))
    idx_list = [0, idx_train, idx_val, num_samples]
    
    for i, category in enumerate(['train', 'val', 'test']):
        df_ind = pd.DataFrame(columns=['sample_name'])
        df_ind['sample_name'] = ['Graph'+str(fileidx)+'.pt' for fileidx in range(idx_list[i], idx_list[i+1])]
        indexFile = os.path.join(processed_dir, '{}_index.csv'.format(category))
        df_ind.to_csv(indexFile)
        if dataset_name == 'AIR_BJ' or 'AIR_GZ':
            time_interval = 60
        elif dataset_name == 'PEMS08' or 'WaterQuality':
            time_interval = 5
        processed[category] = Dataset_CaST_processed(processed_dir, indexFile, in_seq=seq_length_x, time_interval=time_interval)

    results['train_loader'] = torch_geometric.loader.DataLoader(processed['train'], batch_size, shuffle=True)
    results['val_loader'] = torch_geometric.loader.DataLoader(processed['val'], batch_size, shuffle=False)
    results['test_loader'] = torch_geometric.loader.DataLoader(processed['test'], batch_size, shuffle=False)

    print('train: {}\t valid: {}\t test:{}'.format(len(results['train_loader'].dataset),
                                                   len(results['val_loader'].dataset),
                                                   len(results['test_loader'].dataset)))
    results['scalers'] = scalers
    return results

def check_device(device=None):
    if device is None:
        print("`device` is missing, try to train and evaluate the model on default device.")
        if torch.cuda.is_available():
            print("cuda device is available, place the model on the device.")
            return torch.device("cuda")
        else:
            print("cuda device is not available, place the model on cpu.")
            return torch.device("cpu")
    else:
        if isinstance(device, torch.device):
            return device
        else:
            return torch.device(device)

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)

def get_num_nodes(dataset):
    print(dataset)
    d = {'AIR_BJ': 34,  
        'AIR_GZ': 41,
        'PEMS08':170}
    assert dataset[:6] in d.keys()
    return d[dataset[:6]]

def get_num_edges(dataset):
    d = {'AIR_BJ': 82,  
        'AIR_GZ': 77,
        'PEMS08':303}
    assert dataset[:6] in d.keys()
    return d[dataset[:6]]

def get_null_value(dataset):
    d = {'AIR_BJ': 0.0,  
        'AIR_GZ': 0.0,
        'PEMS08': 0.0 }
    assert dataset[:6] in d.keys()
    return d[dataset[:6]]




