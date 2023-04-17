import torch
import torch.nn as nn
import json
import numpy as np

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
    def __init__(self,
                 args, models, data2model):
        super(Ensemble, self).__init__()
        init_w = torch.ones(args.num_users, args.num_models) / args.num_models
        self.embedding = nn.Embedding(args.num_users, args.num_models)
        self.embedding.weight = torch.nn.Parameter(init_w)

        self.linears = nn.Linear(in_features = args.num_models, 1)
        self._initialize_weights()
        self.num_models = args.num_models

        self.models = models
        self.data2model = data2model
        
    def forward(self, inputs):
        pdb.set_trace()
        user_emb = self.embedding(user_index)
        predict = [self.models[i](inputs[i]) for i in range(self.num_models)]
        
        return np.dot(user_emb, predict)
  