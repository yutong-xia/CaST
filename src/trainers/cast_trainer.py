import logging
import os
import time
from typing import Optional, List, Union

import numpy as np
import pandas as pd
import torch
from torch import nn, Tensor
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import RMSprop
import torch.nn.functional as F

from src.utils.logging import get_logger
from src.base.trainer import BaseTrainer
from src.utils import graph_algo
from src.utils.metrics import masked_rmse
from src.utils import metrics as mc
import csv
import pickle as pkl

class CaST_Trainer(BaseTrainer):
    def __init__(self,  
                 beta1,
                 beta2,
                 **args):
        super(CaST_Trainer, self).__init__(**args)
        self.beta1 = beta1
        self.beta2 = beta2
        self.mi_regulization = nn.CrossEntropyLoss()

    def _calculate_supports(self, adj_mat, filter_type):

        num_nodes = adj_mat.shape[0]
        new_adj = adj_mat + np.eye(num_nodes)

        if filter_type == "identity":
            supports = np.diag(np.ones(new_adj.shape[0])).astype(np.float32)
            supports = Tensor(supports).cuda()
        else:
            scaled_adj = graph_algo.calculate_scaled_laplacian(new_adj).todense()
            cheb_poly_adj = graph_algo.calculate_cheb_poly(scaled_adj, 3)
            supports = Tensor(cheb_poly_adj).cuda()
        return supports
    

    def train_batch(self, X, label, iter):
        self.optimizer.zero_grad()
        label = label.squeeze(1) 

        pred, h_node_env, h_node_env_q, env_ind, env_cla_pred= self.model(X)
        pred, label = self._inverse_transform([pred, label])
        
        # prediction loss
        loss_pred = self.loss_fn(pred, label, 0.0)
        # Vector quantization objective
        loss_vq = F.mse_loss(h_node_env,h_node_env_q)
        # Commitment objective
        loss_commit = F.mse_loss(h_node_env_q, h_node_env)
        # mutual info for env and entity
        loss_mi = - self.mi_regulization(env_cla_pred, env_ind)
        
        # total loss
        loss = loss_pred + loss_vq + self.beta1 * loss_commit + self.beta2 * loss_mi
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self._clip_grad_value)
        self.optimizer.step()
        return loss.item()

    def train(self):
        self.logger.info("start training !!!!!")
        # training phase
        iter = 0
        val_losses = [np.inf]
        saved_epoch = -1
        train_losses_list = []
        eval_losses_list = []
        for epoch in range(self._max_epochs):
            self.model.train()
            train_losses = []
            if epoch - saved_epoch > self._patience:
                self.early_stop(epoch, min(val_losses))
                break
            start_time = time.time()
            for i, data in enumerate(self.data['train_loader']):
                (X, label) = data
                X, label = self._check_device([X, label])
                train_losses.append(self.train_batch(X, label, iter))
                iter += 1
                if iter != None:
                    if iter % self._save_iter == 0:
                        val_loss = self.evaluate()
                        message = 'Epoch [{}/{}] ({}) train_mae: {:.4f}, val_mae: {:.4f} '.format(epoch,
                                    self._max_epochs,
                                    iter,
                                    np.mean(train_losses),
                                    val_loss)
                        self.logger.info(message)

                        if val_loss < np.min(val_losses):
                            model_file_name = self.save_model(
                                epoch, self._save_path, self._n_exp)
                            self._logger.info(
                                'Val loss decrease from {:.4f} to {:.4f}, '
                                'saving to {}'.format(np.min(val_losses), val_loss, model_file_name))
                            val_losses.append(val_loss)
                            saved_epoch = epoch
                            
                        train_losses_list.append(np.mean(train_losses))
                        eval_losses_list.append(val_loss)
                            
            end_time = time.time()
            self.logger.info("epoch complete")
            self.logger.info("evaluating now!")

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            val_loss = self.evaluate()

            if self.lr_scheduler is None:
                new_lr = self._base_lr
            else:
                new_lr = self.lr_scheduler.get_last_lr()[0]

            message = 'Epoch [{}/{}] ({}) train_mae: {:.4f}, val_mae: {:.4f}, lr: {:.6f}, ' \
                '{:.1f}s'.format(epoch,
                                 self._max_epochs,
                                 iter,
                                 np.mean(train_losses),
                                 val_loss,
                                 new_lr,
                                 (end_time - start_time))
            self._logger.info(message)

            if val_loss < np.min(val_losses):
                model_file_name = self.save_model(
                    epoch, self._save_path, self._n_exp)
                self._logger.info(
                    'Val loss decrease from {:.4f} to {:.4f}, '
                    'saving to {}'.format(np.min(val_losses), val_loss, model_file_name))
                val_losses.append(val_loss)
                saved_epoch = epoch
    
    def test_batch(self, X, label, mode='eval'):
        label = label.squeeze(1) 
        pred, _, _, env_ind, _ = self.model(X, test_flag=True)
        pred, label = self._inverse_transform([pred, label])
        if mode=='test':
            return pred, label, env_ind
        else:
            return pred, label

    def test(self, epoch, mode='test'):
        self.load_model(epoch, self.save_path, self._n_exp)

        labels = []
        preds = []
        env_inds = []
        
        start_time = time.time()
        
        with torch.no_grad():
            self.model.eval()
            for _, data in enumerate(self.data[mode + '_loader']):
                (X, label) = data
                X, label = self._check_device([X, label])
                pred, label, env_ind = self.test_batch(X, label, mode)
                labels.append(label.cpu())
                preds.append(pred.cpu())
                env_inds.append(env_ind.cpu())

        end_time = time.time()
        
        labels = torch.cat(labels, dim=0)
        preds = torch.cat(preds, dim=0)
         
        amae = []
        amape = []
        armse = []

        for i in range(self.model.horizon):
            pred = preds[:, i]
            real = labels[:, i]
            metrics = mc.compute_all_metrics(pred, real, self.null_value )
            log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}, teat_time {:.1f}s'
            print(log.format(i+1, metrics[0], metrics[1], metrics[2], (end_time - start_time)))
            amae.append(metrics[0])
            amape.append(metrics[1])
            armse.append(metrics[2])

        log = 'On average over {} horizons, Average Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(self.model.horizon, np.mean(amae), np.mean(amape), np.mean(armse)))
        
        ###### get each periods performance
        interval = self.model.horizon//3
        amae_day = []
        amape_day = []
        armse_day = []

        for i in range(0, self.model.horizon, interval):
            pred = preds[:, i: i + interval]
            real = labels[:, i: i + interval]
            metrics = mc.compute_all_metrics(pred, real, self.null_value)
            amae_day.append(metrics[0])
            amape_day.append(metrics[1])
            armse_day.append(metrics[2])

        log = '0-7 (1-8h) Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(amae_day[0], amape_day[0], armse_day[0]))
        log = '8-15 (9-16h) Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(amae_day[1], amape_day[1], armse_day[1]))
        log = '16-23 (17-24h) Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(amae_day[2], amape_day[2], armse_day[2]))
            
        csv_path  = self.result_path + '/{}.csv'.format(self.model_name)
        if not os.path.exists(csv_path):
            df = pd.DataFrame(columns = ['hp','end_time','time',
                                            'mae','mape','rmse',
                                            'mae_1','mape_1','rmse_1',
                                            'mae_2','mape_2','rmse_2',
                                            'mae_3','mape_3','rmse_3'])
            df.to_csv(csv_path, index = False)
            
        with open(csv_path,'a+') as f:
            csv_write = csv.writer(f)
            data_row = [self.hp, time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()), round(end_time - start_time, 2),
                        np.mean(amae), np.mean(amape), np.mean(armse),
                        amae_day[0], amape_day[0], armse_day[0],
                        amae_day[1], amape_day[1], armse_day[1],
                        amae_day[2], amape_day[2], armse_day[2]
                        ]
            csv_write.writerow(data_row)
            
        return np.mean(amae)
    