import torch
import torch.nn as nn
import json
import numpy as np
class DeepFM(nn.Module):
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
                 args, data):
        super(DeepFM, self).__init__()
        
        self.field_dims = data['field_dims']
        self.input_dim = sum(self.field_dims)
        self.num_fields = len(self.field_dims)

        self.encoding_dims = np.concatenate([[0], np.cumsum(self.field_dims)[:-1]])
        self.factor_dim = args.factor_dim
        self.dnn_hidden_units= [args.dnn_hidden_units] * 3
        self.drop_out = args.dropout
        self.dnn_activation = args.activation
        self.dnn_use_bn = args.dnn_use_bn

        
        self.embedding = nn.ModuleList([
            nn.Embedding(feature_size, self.factor_dim) for feature_size in self.field_dims
        ])
        
        
        self.fm = FMLayer(input_dim=self.input_dim)
        self.dnn = DNNLayer(input_dim=(self.num_fields * self.factor_dim), 
                            hidden_units=self.dnn_hidden_units, 
                            activation=self.dnn_activation, 
                            dropout_rate=self.drop_out, use_bn=self.dnn_use_bn)
        
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)
        
                
    def forward(self, x):
        '''
        Parameter
            x: Long tensor of size "(batch_size, num_fields)"
                sparse_x : Same with `x_multihot` in FieldAwareFM class
                dense_x  : Similar with `xv` in FFMLayer class. 
                           List of "num_fields" float tensors of size "(batch_size, factor_dim)"
        Return
            y: Float tensor of size "(batch_size)"
        '''
        
        sparse_x = x + x.new_tensor(self.encoding_dims).unsqueeze(0)
        sparse_x = torch.zeros(x.size(0), self.input_dim, device=x.device).scatter_(1, x, 1.)
        dense_x = [self.embedding[f](x[...,f]) for f in range(self.num_fields)] 
        
        y_fm = self.fm(sparse_x, torch.stack(dense_x, dim=1))
        y_dnn = self.dnn(torch.cat(dense_x, dim=1))
        
        
        y = y_fm + y_dnn.squeeze(1)

        return y
    
class FMLayer(nn.Module):
    def __init__(self, input_dim):
        '''
        Parameter
            input_dim: Entire dimension of input vector (sparse)
            factor_dim: Factorization dimension
        '''
        super().__init__()
        self.input_dim = input_dim
        self.linear = nn.Linear(self.input_dim, 1, bias=True) # FILL HERE : Fill in the places `None` #
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                #nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def square(self, x):
        return torch.pow(x,2)

    def forward(self, sparse_x, dense_x):
        '''
        Parameter
            sparse_x : Same with `x_multihot` in FieldAwareFM class
                       Float tensor with size "(batch_size, self.input_dim)"
            dense_x  : Similar with `xv` in FFMLayer class. 
                       Float tensors of size "(batch_size, num_fields, factor_dim)"
        
        Return
            y: Float tensor of size "(batch_size)"
        '''
        y_linear = self.linear(sparse_x)
        
        square_of_sum = self.square(torch.sum(dense_x, dim=1))
        sum_of_square = torch.sum(self.square(dense_x), dim=1)
        y_pairwise = 0.5 * torch.sum(square_of_sum - sum_of_square, dim=1)
        
        y_fm = y_linear.squeeze(1) + y_pairwise

        return y_fm
        
        
def activation_layer(act_name):
    '''Select activation layer by its name
    Parameter
        act_name: String value or nn.Module, name of activation function
    Return
        act_layer: Activation layer
    '''
    if isinstance(act_name, str):
        if act_name.lower() == 'sigmoid':
            act_layer = nn.Sigmoid()
        elif act_name.lower() == 'linear':
            act_layer = nn.Identity()
        elif act_name.lower() == 'relu':
            act_layer = nn.ReLU(inplace=True)
        elif act_name.lower() == 'prelu':
            act_layer = nn.PReLU()
    elif issubclass(act_name, nn.Module):
        act_layer = act_name()
    else:
        raise NotImplementedError

    return act_layer

class DNNLayer(nn.Module):
    '''The Multi Layer Percetron (MLP); Fully-Connected Layer (FC); Deep Neural Network (DNN) with 1-dimensional output
    Parameter
        inputs_dim: Input feature dimension
        hidden_units: List of positive integer, the layer number and units in each layer
        dropout_rate: Float value in [0,1). Fraction of the units to dropout
        activation: Activation function to use
        use_bn: Boolean value. Whether use BatchNormalization before activation
    ''' 
    def __init__(self, 
                 input_dim, 
                 hidden_units, 
                 dropout_rate=0, 
                 activation='relu', 
                 use_bn=False,
                 **kwargs):
        super().__init__()
        self.hidden_units = hidden_units
        self.activation = activation
        self.dropout = nn.Dropout(dropout_rate)
        self.use_bn = use_bn
        
        layer_size = len(hidden_units)
        hidden_units = [input_dim] + list(hidden_units)

        self.linears = nn.ModuleList([
            nn.Linear(hidden_units[i], hidden_units[i+1]) for i in range(layer_size)
        ])

        if self.use_bn:
            self.bn = nn.ModuleList([
                nn.BatchNorm1d(hidden_units[i+1]) for i in range(layer_size)
            ])

        self.activation_layers = nn.ModuleList([
            activation_layer(activation) for i in range(layer_size)
        ])
        
        self.dnn_linear = nn.Linear(hidden_units[-1], 1, bias=False)
        
        self._initialize_weights()
        
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                #nn.init.normal_(m.weight, 0, 0.01)
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        '''
        Parameter
            x: nD tensor of size "(batch_size, ..., input_dim)"
               The most common situation would be a 2D input with shape "(batch_size, input_dim)".
        
        Return
            y: nD tensor of size "(batch_size, ..., 1)"
               For instance, if input x is 2D tensor, the output y would have shape "(batch_size, 1)".
        '''
        deep_input = x
        for i in range(len(self.linears)):
            fc = self.linears[i](deep_input)
            if self.use_bn:
                fc = self.bn[i](fc)
            fc = self.activation_layers[i](fc)
            fc = self.dropout(fc)
            deep_input = fc
        
        y_dnn = self.dnn_linear(deep_input)
            
        return y_dnn