import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
import pdb
class Ensemble(nn.Module):
    '''The DeepFM architecture
    Parameter
        field_dims: List of field dimensions
        factor_dim: Factorization dimension for dense embedding
        dnn_hidden_units: List of positive integer, the layer number and units in each layer
        dnn_dropout: Float value in [0,1). Fraction of the units to dropout in DNN layer
        dnn_activation: Activation function to use in DNN layer
        dnn_use_bn: Boolean value. Whether use BatchNormalization before activation in DNN layer
    '''
    def __init__(self,args):
        super(Ensemble, self).__init__()
        init_w = torch.ones(args.num_users, args.num_models,dtype= torch.float32) / args.num_models
        self.embedding = nn.Embedding(args.num_users, args.num_models)
        self.embedding.weight = torch.nn.Parameter(init_w)
        self.embedding = self.embedding.to(args.device)
        #self.linears = nn.Linear(in_features = args.num_models, out_features =  1)
        self.num_models = args.num_models

        #self.data2model = data2model
        
    def forward(self, inputs, user_idx):
        user_emb = self.embedding(user_idx).squeeze()
        inputs = inputs.squeeze()
        #predict = self.linears(predict)
        
        return torch.dot(user_emb, inputs)


class Ensemble2(nn.Module):
    '''The DeepFM architecture
    Parameter
        field_dims: List of field dimensions
        factor_dim: Factorization dimension for dense embedding
        dnn_hidden_units: List of positive integer, the layer number and units in each layer
        dnn_dropout: Float value in [0,1). Fraction of the units to dropout in DNN layer
        dnn_activation: Activation function to use in DNN layer
        dnn_use_bn: Boolean value. Whether use BatchNormalization before activation in DNN layer
    '''
    def __init__(self,args):
        super(Ensemble2, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(in_features = args.num_models, out_features = args.num_models),
            nn.Linear(args.num_models, 1),
            nn.ReLU()
        )
        
        

        #self.data2model = data2model
        
    def forward(self, x):
        return self.linear(x)
  