
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.base.model import BaseModel
from src.layers.cell import *
from src.layers.cast_cell import *
from src.layers.dilated_conv import DilatedConvEncoder
import math

class CaST(BaseModel):
    def __init__(self,
                 num_edges,
                 dropout=0.3,
                 hid_dim = 16,
                 node_embed_dim = 20,
                 K = 3,
                 depth = 10,
                 bias = True,
                 n_envs = 10,
                 time_delay_scaler = 6,
                 **args):
        super(CaST, self).__init__(**args)
        self.dropout = dropout
        self.hid_dim = hid_dim
        self.node_embed_dim = node_embed_dim
        self.n_node = self.num_nodes
        self.n_edge = num_edges
        self.n_envs = n_envs
        self.K = K
        self.start_encoder = DilatedConvEncoder(in_channels=self.input_dim, channels=[self.input_dim] * depth + [self.hid_dim], kernel_size=3)
        
        ############ temperoal ############
        # seperate the temporal part into two parts: entity and environment
        t_kernels = [2**i for i in range(int(math.log2(self.seq_len//2)))]
        self.temporal = TempDisentangler(input_dims =self.hid_dim, 
                                         output_dims =self.hid_dim*2,
                                        kernels = t_kernels,
                                        length = self.seq_len,
                                        hidden_dims=self.hid_dim,
                                        depth= depth,
                                        dropout = dropout)
        ###### envrionment ########
        # codebook for environment
        self.codebook = EnvEmbedding(n_envs, node_embed_dim * self.seq_len)
        # project the environment to fit the environment codebook
        self.t_proj_env = nn.Linear(self.hid_dim, node_embed_dim)
        # shape the environment to fit the node representation
        self.env_lin = nn.Linear(node_embed_dim * self.seq_len, hid_dim)

        ###### entity ########
        # reduce the time dimension of entity
        self.t_proj_cau = nn.Linear(self.seq_len, 1)

        ########## edge features ###########
        # Start MLP for edges
        self.start_mlp_edge = nn.Linear(2 + self.seq_len//time_delay_scaler, hid_dim)
        # HodgeLaguerreConv for edges
        self.spatial_edge = HodgeLaguerreConv(in_channels=hid_dim, out_channels=hid_dim, K=K, bias=bias)
        # project the updated edge features to the causal score
        self.edge_causal = nn.Linear(hid_dim, K*2)

        ######### message passing ###########
        self.spatial_node = GCNConv(in_channels=hid_dim, num_nodes=self.n_node, out_channels=hid_dim, K=K)

        ########## mutual info regulization #########
        self.env_cla = nn.Sequential(nn.Linear(hid_dim,hid_dim*2),
                                    nn.ReLU(),
                                    nn.Dropout(dropout),
                                    nn.Linear(hid_dim*2, n_envs),
                                    nn.Softmax(dim = 1)
                                    )

        ############ node_embedding ##################
        self.node_embed = nn.Parameter(torch.randn(self.n_node, node_embed_dim), requires_grad=True)
        self.node_embed_lin_ent = nn.Linear(node_embed_dim, self.hid_dim)
        self.node_embed_lin_env = nn.Linear(node_embed_dim, self.hid_dim)

        ############## predictor ###########
        self.end_mlp = nn.Sequential(nn.LayerNorm([self.n_node, hid_dim * 2]),
                                    nn.Linear(hid_dim * 2, 256),
                                    nn.ReLU(),
                                    nn.Dropout(dropout),
                                    nn.Linear(256, 512),
                                    nn.ReLU(),
                                    nn.Dropout(dropout),
                                    nn.Linear(512, self.horizon * self.output_dim),
                                    )

    def forward(self, X, test_flag=False):
        '''
        input:  #### edge #####
                graph.x_s                [batch_size * num_edges, edge_feature_dim]
                graph.edge_index_s       [2, num_edge_edge] 
                graph.edge_weight_s      [num_edge_edge]

                #### node #####
                graph.x_t                [batch_size * num_nodes, in_seq, indim]
                graph.edge_index_t       [2, num_edges]
                graph.edge_weight_t      [num_edges]
        '''
        # get the input
        x_link, edge_index_link, edge_weight_link = X.x_s, X.edge_index_s, X.edge_weight_s # edge
        x_node, edge_index_node, edge_weight_node = X.x_t, X.edge_index_t, X.edge_weight_t # node
        edge_index = X.edge_index 

        # get the shape
        b, l, d = x_node.shape
        batch_size = b//self.n_node
        
        # project the input into latent space
        h_node = self.start_encoder(x_node.float().permute(0,2,1)) 

        # sperate the temporal part into entity and environment
        h_environment, h_entity = self.temporal(h_node)
        
        ############ enviroment ############
        # project the environment to fit the environment codebook
        h_environment = self.t_proj_env(h_environment).reshape(-1, self.seq_len * self.node_embed_dim)
        # find the most similar environment in the codebook
        if not test_flag: 
            env_output, env_q, env_ind = self.codebook.straight_through(h_environment)
        else: 
            env_output, env_q, env_ind = self.codebook.straight_through_test(h_environment)
            env_ind = env_ind.reshape(batch_size,self.n_node,self.n_envs)
        env_output = self.env_lin(env_output).reshape(batch_size, self.n_node, -1)
            
        ############# edge feature to causal score ############
        # update the edge feature to recogenize the causal score 
        h_link = self.start_mlp_edge(x_link.float())
        h_link_updated = self.spatial_edge(h_link, edge_index_link, edge_weight_link)
        norm_causal_score = self.edge_causal(h_link_updated)

        ############ entity and message passing ############
        # reduce the time dimension
        h_entity = self.t_proj_cau(h_entity.permute(0,2,1)).squeeze() 
        # update the node representation based on the causal score
        h_entity = self.spatial_node(h_entity, edge_index, norm_causal_score) 

        ############ node embedding ############
        node_embed_ent = self.node_embed_lin_ent(self.node_embed)
        node_embed_env = self.node_embed_lin_env(self.node_embed)
        # add to entity
        h_entity = torch.add(h_entity, node_embed_ent.expand(batch_size,-1,-1))
        # add to environment
        env_output = torch.add(env_output, node_embed_env.expand(batch_size,-1,-1))

        ## get the final node representation
        h_final_repr = torch.cat([env_output, h_entity],dim=-1) 
        
        # do prediction
        pred = self.end_mlp(h_final_repr).permute(0, 2, 1).reshape(batch_size,l,self.n_node,self.output_dim)

        ## get the mutual information regulization
        env_cla_pred = self.env_cla(h_entity.reshape(batch_size * self.n_node, -1))

        return pred, h_environment, env_q, env_ind, env_cla_pred