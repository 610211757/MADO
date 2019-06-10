# coding:utf-8

import warnings
warnings.filterwarnings("ignore")


import torch
import torch.nn as nn

from torch.nn.parameter import Parameter

import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import math

import torch
import random

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(423)

from sklearn.metrics import mean_squared_error

"""

因为view需要tensor的内存是整块的 

contiguous：view只能用在contiguous的variable上。
如果在view之前用了transp5ose, permute等，需要用contiguous()来返回一个contiguous copy。 

一种可能的解释是： 
有些tensor并不是占用一整块内存，而是由不同的数据块组成，而tensor的view()操作依赖于内存是整块的，
这时只需要执行contiguous()这个函数，把tensor变成在内存中连续分布的形式。 

"""



class MADO_Net(nn.Module):

    def __init__(self, config, seq_len, fea_dim, mode):

        super(MADO_Net, self).__init__()

        self.objective = 'one-class'
        self.c = None
        self.nu = 0.1  # nu: Deep SVDD hyperparameter nu (must be 0 < nu <= 1).
        self.mode = mode
        self.add_lstm_prediction = config.add_lstm_prediction
        self.weight_lstm_prediction = config.weight_lstm_prediction

        self.config = config

        self.input_dim = fea_dim
        self.input_dim_lstm_gcn = fea_dim + config.gcn_out_dim
        self.allow_gcn_to_lstm = config.allow_gcn_to_lstm

        self.lstm_hidden_dim = config.lstm_hidden_dim
        self.gcn_hidden_dim = config.gcn_hidden_dim
        self.seq_len = seq_len

        self.gcn_out_dim = config.gcn_out_dim
        self.global_out_dim = config.global_out_dim
        self.local_out_dim = config.local_out_dim
        self.final_out_dim = config.final_out_dim
        self.rep_dim = self.final_out_dim
        self.lstm_prediction_dim = self.input_dim
        self.mse_loss_fn = torch.nn.MSELoss()

        self.cnn_out_dim = config.cnn_out_dim

        if self.config.dataset_name == "wadi":
            self.filters = [3, 16, 32]
        elif self.config.dataset_name == "kdd99":
            self.filters = [3, 12, 32]
        elif self.config.dataset_name == "swat":
            self.filters = [3, 16, 66]

        self.dropout = config.dropout

        if config.allow_gcn_to_lstm is True:
            self.lstm = nn.LSTM(self.input_dim_lstm_gcn, self.lstm_hidden_dim, num_layers=config.nums_lstm_layer, batch_first=True) 
        else:
            self.lstm = nn.LSTM(self.input_dim, self.lstm_hidden_dim, num_layers=config.nums_lstm_layer, batch_first=True)

        # self.fc_global = nn.Linear(self.lstm_hidden_dim * self.seq_len, self.global_out_dim, bias=False)
        self.fc_local = nn.Linear(self.lstm_hidden_dim, self.local_out_dim, bias=False)
        self.fc_lstm_prediction = nn.Linear(self.lstm_hidden_dim, self.lstm_prediction_dim, bias=False)

        # graph CNN
        self.gc1 = GraphConvolution(self.seq_len, self.gcn_hidden_dim)
        self.gc2 = GraphConvolution(self.gcn_hidden_dim, self.gcn_hidden_dim)
        self.fc_graph = nn.Linear(self.input_dim * self.gcn_hidden_dim, self.gcn_out_dim, bias=False)

        ## global CNN componet
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_channels=self.lstm_hidden_dim,
                                    out_channels=self.cnn_out_dim ,
                                    kernel_size=h),
                          # nn.BatchNorm1d(num_features=config.feature_size),
                          nn.ReLU(),
                          nn.AvgPool1d(kernel_size=self.seq_len - h + 1))  # AvgPool1d, MaxPool1d
            for h in self.filters
        ])
        self.fc_global = nn.Linear(in_features= self.cnn_out_dim * len(self.filters), out_features=self.global_out_dim)

        if self.mode == "lstm-cnn-gcn-one-class":
            self.one_class_rep_dim = self.gcn_out_dim+self.global_out_dim+self.local_out_dim
        elif self.mode == "lstm-cnn-one-class":
            self.one_class_rep_dim = self.global_out_dim+self.local_out_dim
        elif self.mode == "lstm-gcn-one-class":
            self.one_class_rep_dim = self.gcn_out_dim+self.local_out_dim
        elif self.mode == "lstm-one-class":
            self.one_class_rep_dim = self.local_out_dim

        self.fc_final = nn.Linear(self.one_class_rep_dim , self.final_out_dim, bias=False)

        self.add_learned_r = config.add_learned_r
        if self.add_learned_r:
            self.fc_adjust_r = nn.Linear(self.global_out_dim , 1, bias=False)


    def forward(self, x, adj):
        
        # graph CNN
        x_transpose = torch.transpose(x, 1, 2)
        graph_x = F.relu(self.gc1(x_transpose, adj))
        graph_x = F.dropout(graph_x, self.dropout)
        graph_x = self.gc2(graph_x, adj)  
        graph_x = graph_x.contiguous().view(graph_x.size(0), -1)
        graph_x = self.fc_graph(graph_x)
        graph_x = F.relu(graph_x)
        graph_x = F.dropout(graph_x, self.dropout)

        # np.shape(graph_x)  256, 16
        # np.shape(x)   [256, 120, 34])
        # lstm_in_graph_x [256, 120, 16]

        if self.allow_gcn_to_lstm is True:
            lstm_in_graph_x_temp = torch.unsqueeze(graph_x, dim=1)
            lstm_in_graph_x = [ lstm_in_graph_x_temp[i].repeat(np.shape(x)[1], 1) for i, v in enumerate(x)]
            lstm_in_graph_x = torch.stack(lstm_in_graph_x)
            lstm_input = torch.cat([x, lstm_in_graph_x], 2)  # torch.Size([64, 120, 50]
        else:
            lstm_input = x

        # out: 256, 120, 128
        out, _ = self.lstm(lstm_input)
        # lstm_out = out.contiguous().view(out.size(0), -1)
        # lstm_out = self.fc_global(lstm_out)
        # lstm_out = F.leaky_relu(lstm_out)
        # lstm_out = F.dropout(lstm_out, self.dropout)

        # global CNN
        # batch_size x text_len x embedding_size  -> batch_size x embedding_size x text_len
        out_permute = out.permute(0, 2, 1)
        out_CNN = [conv(out_permute) for conv in self.convs]
        out_CNN_cat = torch.cat(out_CNN, dim=1)
        out_CNN_cat = out_CNN_cat.view(-1, out_CNN_cat.size(1))
        out_CNN_cat = self.fc_global(out_CNN_cat)

        if self.add_learned_r:
            self.adjusted_r = self.fc_adjust_r(out_CNN_cat)**2

        out_all = []

        if self.add_lstm_prediction:
            self.loss_prediction=0

        for i in range(self.seq_len):
            per_out = self.fc_local(out[:, i, :])
            if self.mode == "lstm-cnn-gcn-one-class":
                cat_out = torch.cat([graph_x, per_out, out_CNN_cat], 1)
                cat_out = F.leaky_relu(cat_out)
                cat_out = F.leaky_relu(self.fc_final(cat_out))
                cat_out = F.dropout(cat_out, self.dropout)
            elif self.mode == "lstm-cnn-one-class":
                cat_out = torch.cat([per_out, out_CNN_cat], 1)   # 没有GCN
                cat_out = F.leaky_relu(cat_out)
                cat_out = F.leaky_relu(self.fc_final(cat_out))
                cat_out = F.dropout(cat_out, self.dropout)
            elif self.mode == "lstm-gcn-one-class":
                cat_out = torch.cat([graph_x, per_out], 1)
                cat_out = F.leaky_relu(cat_out)           # 没有 CNN         
                cat_out = F.leaky_relu(self.fc_final(cat_out))
                cat_out = F.dropout(cat_out, self.dropout)
            elif self.mode == "lstm-one-class":
                cat_out = per_out
                cat_out = F.leaky_relu(cat_out)           # 没有GCN 没有 CNN         
                cat_out = F.leaky_relu(self.fc_final(cat_out))
                cat_out = F.dropout(cat_out, self.dropout)
            out_all.append(cat_out)

            if self.add_lstm_prediction:
                
                if i <= (self.seq_len-2):
                    # next_prediction 256, 34
                    next_prediction = self.fc_lstm_prediction(out[:, i, :])
            
                    self.loss_prediction += torch.sqrt(self.mse_loss_fn(next_prediction, x[:, i+1, :]))
        return out_all

    def computing_oneclass_loss(self, config, outputs):

        # assert( self.seq_len == len(outputs) )

        loss = 0

        if self.add_learned_r:

            dim_expand_r = outputs[0].shape[0]
            dist_0 = torch.zeros(dim_expand_r, 1).to(config.device)
            
        for i in range(self.seq_len):
            if self.add_learned_r:
                # print("haha", np.shape(outputs[i])) # [64, 32] batchsize 64
                dist = torch.sum((outputs[i] - self.c) ** 2, dim=1) 
                dist = torch.unsqueeze(dist, dim =1)
                dist = dist - self.adjusted_r
                
                dist = torch.cat([dist_0, dist], dim = 1)
                dist = torch.max(dist, 1)[0]
                # if self.config.dataset_name == "wadi":
                #     dist = dist + 10000 * self.adjusted_r
                # elif self.config.dataset_name == "kdd99":
                #     dist = dist + 100 * self.adjusted_r
                # elif self.config.dataset_name == "swat":
                #     dist = dist + 10000 * self.adjusted_r
                dist = dist + config.v * self.adjusted_r
            else:
                dist = torch.sum((outputs[i] - self.c) ** 2, dim=1)
            loss += torch.mean(dist)
        loss = loss/self.seq_len

        if self.add_lstm_prediction:
            loss+=self.weight_lstm_prediction*self.loss_prediction

        return loss
    

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'



