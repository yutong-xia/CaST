from torch_geometric.data import Dataset, Data
from torch_geometric.utils import dense_to_sparse
import torch
import numpy as np

from torch import Tensor
from scipy import sparse
from torch_sparse import SparseTensor
import os
from dtaidistance import dtw
import pandas as pd

# Reference code: https://github.com/JH-415/HL-HGAT

class PairData(Data):
    def __init__(self, edge_index_s=None, x_s=None, edge_index_t=None, x_t=None,
                edge_weight_s=None, edge_weight_t=None, y=None):
        super().__init__()
        self.edge_index_s = edge_index_s
        self.x_s = x_s
        self.edge_index_t = edge_index_t
        self.x_t = x_t
        self.edge_weight_s = edge_weight_s
        self.edge_weight_t = edge_weight_t
        self.y = y

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_s':
            return self.x_s.size(0)
        if key == 'edge_index_t':
            return self.x_t.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)
        

## Dataset           
class Dataset_CaST(Dataset):
    def __init__(self, rootDir, adjFile, scalers, input_dim, output_dim, seq_length_x, seq_length_y, interval, time_delay_scaler):
        self.data = np.load(rootDir) #(b,n,d)
        self.time_stamps = self.data.shape[0]
        self.n_nodes = self.data.shape[1]
        
        # adj
        self.dist_adj = np.load(os.path.join(adjFile,'dist_adj.npy')) 
        self.sema_adj = np.load(os.path.join(adjFile,'peacor_adj.npy'))
        self.sparse_adj = np.load(os.path.join(adjFile,'sparse_adj.npy')) 
        
        # edge
        self.time_delay_scaler = time_delay_scaler
        self.time_dalay_attr_file = os.path.join(adjFile,'time_dalay_attr.pkl')

        # norm
        self.scalers = scalers
        
        # para
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seq_length_x = seq_length_x
        self.seq_length_y = seq_length_y
        self.interval = interval
        
        self.x_offsets = np.sort(np.concatenate((np.arange(-(self.seq_length_x - 1) * self.interval, 1, self.interval),)))
        self.y_offsets = np.sort(np.arange(self.interval, (self.seq_length_y + 1) * self.interval, self.interval))
        
        
        self._norm()
        self._get_edge_attr() # get the edge_index and edge_attr
        
    def _get_edge_attr(self):
        '''
        get edge features from distance-based adj and semantic adj
        '''  
        # use sparse_adj to generate index
        adj_tensor = SparseTensor.from_dense(torch.from_numpy(self.sparse_adj))
        row, col, _ = adj_tensor.t().coo()
        edge_index = torch.stack([row, col], dim=0)

        # use dist_adj and sema_adj to generate attribute
        edge_attr1, edge_attr2 =[], []
        for i in range(edge_index.shape[1]):
            edge_attr1.append(self.dist_adj[edge_index[0,i],edge_index[1,i]])
            edge_attr2.append(self.sema_adj[edge_index[0,i],edge_index[1,i]])
        
        edge_attr1 = np.stack(edge_attr1,0)
        edge_attr2 = np.stack(edge_attr2,0)
        edge_attr = torch.stack([torch.Tensor(edge_attr1), torch.Tensor(edge_attr2)],dim=-1)
        
        # del auto-corr
        idx_ = edge_index[0]< edge_index[1]
        self.edge_index, self.edge_attr = edge_index[:,idx_], edge_attr[idx_]
        
    def _time_delay_similarity(self, ts_i, ts_j):
        '''
        get the time delay similarity between two time series.
        ts_i: [t,1]
        ts_j: [t,1]
        return: [t//scaler]
        '''
        length = ts_i.shape[0]
        sim = []
        for i in range(0,length,self.time_delay_scaler):
            ts_j_new = torch.roll(ts_j, -int(i), 0)
            distance = dtw.distance(ts_i,ts_j_new)
            sim.append(distance)
        sim = np.stack(sim,-1)
        return sim
    
    def _time_delay_similarity_full(self, g):
        '''
        g: a graph with node signal [t, node, dim]
        return: edge signal [n_edge, t//scaler]
        '''
        sim_matrix = []
        for i in range(self.edge_index.shape[1]):
            node_index1, node_index2 = self.edge_index[0,i], self.edge_index[1,i]
            assert node_index1 != node_index2
            sim = self._time_delay_similarity(g[:, node_index1],g[:, node_index2])
            sim_matrix.append(sim)
        sim_matrix = np.stack(sim_matrix,-1)
        return torch.Tensor(sim_matrix).permute(1,0)

    
    def _adj2par(self, edge_index, num_node, num_edge):
        col_idx = torch.cat([torch.arange(edge_index.shape[1]),torch.arange(edge_index.shape[1])]
                            ,dim=-1).to(edge_index.device)
        row_idx = torch.cat([edge_index[0],edge_index[1]], dim=-1).to(edge_index.device)
        val = torch.cat([edge_index[0].new_full(edge_index[0].shape,-1),
                        edge_index[0].new_full(edge_index[0].shape,1)],dim=-1).to(torch.float)
        par_sparse = torch.sparse.FloatTensor(torch.cat([row_idx, col_idx], dim=-1).view(2,-1),
                                            val,torch.Size([num_node, num_edge]))
        return par_sparse
    
    def _norm(self):
        self.data = torch.Tensor(self.data)
        for i in range(self.output_dim):
            self.data[..., i] = self.scalers[i].transform(self.data[..., i])
        
    def __len__(self):
        min_t = abs(min(self.x_offsets))
        max_t = abs(self.time_stamps - abs(max(self.y_offsets))) 
        return int(max_t-min_t)
    
    def __getitem__(self, idx):
        '''
        neighbor: n-1
        '''
        x = self.data[idx + self.x_offsets, :, :self.input_dim] #[t, node, dim]
        y = self.data[idx + self.y_offsets, :, :self.output_dim]
        
        par = self._adj2par(self.edge_index, self.n_nodes, self.edge_index.shape[1]).to_dense()
        L0 = torch.matmul(par, par.T)
        lambda0, _ = torch.linalg.eigh(L0)
        L1 = torch.matmul(par.T, par)
        edge_index_t, edge_weight_t = dense_to_sparse(2*L0/lambda0.max())
        edge_index_s, edge_weight_s = dense_to_sparse(2*L1/lambda0.max())
        
        time_delay_sim = self._time_delay_similarity_full(x)
        x_s = torch.cat([self.edge_attr, time_delay_sim], -1) #[n_edge, 2 + l//scaler]
        x_t = x.permute(1,0,2)
        
        graph = PairData(x_s=x_s, edge_index_s=edge_index_s, edge_weight_s=edge_weight_s, # edge
                            x_t=x_t, edge_index_t=edge_index_t, edge_weight_t=edge_weight_t, # node
                            y = y)
        graph.edge_index = self.edge_index
        graph.num_nodes = self.n_nodes
            
        return [graph,  y]
    

## Dataset
class Dataset_CaST_processed(Dataset):
    def __init__(self, processed_dir, labelFile, in_seq=24, time_interval=60):
        self.processed_path = processed_dir
        self.labelFile = pd.read_csv(labelFile)
        self.in_seq = in_seq
        self.time_stemp_day = 24*60//time_interval
    
    def __len__(self):
        return len(self.labelFile)

    def __getitem__(self,idx):
        file_name = self.labelFile['sample_name'][idx]
        graph_path = os.path.join(self.processed_path, file_name)
        data_zip = torch.load(graph_path)
        graph = data_zip['graph']
        y = data_zip['y']
        return [graph, y]


