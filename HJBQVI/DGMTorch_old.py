import torch
import torch.nn as nn
from utils import MinMaxScaler

class LSTMLayer(nn.Module):
    def __init__(self, output_dim, input_dim, trans1="tanh", trans2="tanh"):
        '''
        Args:
            input_dim (int):       dimensionality of input data
            output_dim (int):      number of outputs for LSTM layers
            trans1, trans2 (str):  activation functions used inside the layer;
                                   one of: "tanh" (default), "relu" or "sigmoid"

        Returns: customized PyTorch layer object used as intermediate layers in DGM
        '''
        super(LSTMLayer, self).__init__()

        # Add properties for layer including activation functions
        self.output_dim = output_dim
        self.input_dim = input_dim

        # Activation function mapping
        activation_map = {
            "tanh": torch.tanh,
            "relu": torch.relu,
            "sigmoid": torch.sigmoid
        }

        self.trans1 = activation_map.get(trans1, torch.tanh)
        self.trans2 = activation_map.get(trans2, torch.tanh)

        # LSTM layer parameters (Xavier initialization)
        # u vectors (weighting vectors for inputs original inputs x)
        self.Uz = nn.Parameter(torch.empty(self.input_dim, self.output_dim))
        self.Ug = nn.Parameter(torch.empty(self.input_dim, self.output_dim))
        self.Ur = nn.Parameter(torch.empty(self.input_dim, self.output_dim))
        self.Uh = nn.Parameter(torch.empty(self.input_dim, self.output_dim))

        # w vectors (weighting vectors for output of previous layer)
        self.Wz = nn.Parameter(torch.empty(self.output_dim, self.output_dim))
        self.Wg = nn.Parameter(torch.empty(self.output_dim, self.output_dim))
        self.Wr = nn.Parameter(torch.empty(self.output_dim, self.output_dim))
        self.Wh = nn.Parameter(torch.empty(self.output_dim, self.output_dim))

        # bias vectors
        self.bz = nn.Parameter(torch.empty(1, self.output_dim))
        self.bg = nn.Parameter(torch.empty(1, self.output_dim))
        self.br = nn.Parameter(torch.empty(1, self.output_dim))
        self.bh = nn.Parameter(torch.empty(1, self.output_dim))

        # Initialize parameters
        self._initialize_weights()

    def _initialize_weights(self):
        # Xavier initialization
        for param in [self.Uz, self.Ug, self.Ur, self.Uh,
                      self.Wz, self.Wg, self.Wr, self.Wh]:
            nn.init.xavier_normal_(param)

        # Initialize biases to zero
        for param in [self.bz, self.bg, self.br, self.bh]:
            nn.init.zeros_(param)

    def forward(self, S, X):
        '''Compute output of a LSTMLayer for a given inputs S,X.

        Args:
            S: output of previous layer
            X: data input

        Returns: customized PyTorch layer output
        '''
        # Compute components of LSTM layer output
        Z = self.trans1(torch.add(torch.add(torch.matmul(X, self.Uz), torch.matmul(S, self.Wz)), self.bz))
        G = self.trans1(torch.add(torch.add(torch.matmul(X, self.Ug), torch.matmul(S, self.Wg)), self.bg))
        R = self.trans1(torch.add(torch.add(torch.matmul(X, self.Ur), torch.matmul(S, self.Wr)), self.br))

        H = self.trans2(torch.add(torch.add(torch.matmul(X, self.Uh), torch.matmul(torch.mul(S, R), self.Wh)), self.bh))

        # Compute LSTM layer output
        S_new = torch.add(torch.mul(torch.sub(torch.ones_like(G), G), H), torch.mul(Z, S))

        return S_new


class DenseLayer(nn.Module):
    def __init__(self, output_dim, input_dim, transformation=None):
        '''
        Args:
            input_dim:       dimensionality of input data
            output_dim:      number of outputs for dense layer
            transformation:  activation function used inside the layer; using
                             None is equivalent to the identity map

        Returns: customized PyTorch (fully connected) layer object
        '''
        super(DenseLayer, self).__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim

        # Dense layer parameters
        self.W = nn.Parameter(torch.empty(self.input_dim, self.output_dim))
        self.b = nn.Parameter(torch.empty(1, self.output_dim))

        # Initialize parameters
        nn.init.xavier_normal_(self.W)
        nn.init.zeros_(self.b)

        # Set transformation function
        if transformation:
            activation_map = {
                "tanh": torch.tanh,
                "relu": torch.relu
            }
            self.transformation = activation_map.get(transformation, None)
        else:
            self.transformation = None

    def forward(self, X):
        '''Compute output of a dense layer for a given input X

        Args:
            X: input to layer
        '''
        # Compute dense layer output
        S = torch.add(torch.matmul(X, self.W), self.b)

        if self.transformation:
            S = self.transformation(S)

        return S


class DGMNet(nn.Module):
    def __init__(self, layer_width, n_layers, input_dim, output_dim = 1, final_trans=None, typeNN = 'LSTM'):
        '''
        Args:
            layer_width:
            n_layers:    number of intermediate LSTM layers
            input_dim:   spatial dimension of input data (EXCLUDES time dimension)
            final_trans: transformation used in final layer

        Returns: customized PyTorch model object representing DGM neural network
        '''
        super(DGMNet, self).__init__()

        # Initial layer as fully connected
        # NOTE: to account for time inputs we use input_dim+1 as the input dimensionality
        self.initial_layer = DenseLayer(layer_width, input_dim+1, transformation="tanh")

        # Intermediate LSTM layers
        self.n_layers = n_layers
        if typeNN == 'LSTM':
            self.LayerList = nn.ModuleList([
                LSTMLayer(layer_width, input_dim+1) for _ in range(self.n_layers)
            ])
        elif typeNN == 'Dense':
            self.LayerList = nn.ModuleList([
                DenseLayer(layer_width, layer_width, transformation="tanh") for _ in range(self.n_layers)
            ])
        elif typeNN == 'Transformer':
            #TODO
            return
        else:
            raise Exception('typeNN ' + typeNN + ' not supported. Choose one of Dense or LSTM')
        self.typeNN = typeNN
        # Final layer as fully connected with a single output (function value)
        self.final_layer = DenseLayer(output_dim, layer_width, transformation=final_trans)

    def forward(self, t, x):
        '''
        Args:
            t: sampled time inputs
            x: sampled space inputs

        Run the DGM model and obtain fitted function value at the inputs (t,x)
        '''
        # Define input vector as time-space pairs

        X = torch.cat([t, x], 1)
        X = MinMaxScaler().fit_transform(X)
        # Call initial layer
        S = self.initial_layer(X)

        # Call intermediate LSTM layers
        for layer in self.LayerList:
            if self.typeNN == 'LSTM':
                S = layer(S, X)
            elif self.typeNN == 'Dense':
                S = layer(S)

        # Call final layer
        result = self.final_layer(S)

        return result

class PIANet(nn.Module):
    def __init__(self, layer_width, n_layers, input_dim, num_classes, final_trans=None, typeNN = 'LSTM'):
        '''
        Args:
            layer_width:
            n_layers:    number of intermediate LSTM layers
            input_dim:   spatial dimension of input data (EXCLUDES time dimension)
            num_classes: number of output classes
            final_trans: transformation used in final layer

        Returns: customized PyTorch model object representing DGM neural network
        '''
        super(PIANet, self).__init__()

        # Initial layer as fully connected
        # NOTE: to account for time inputs we use input_dim+1 as the input dimensionality
        self.initial_layer = DenseLayer(layer_width, input_dim+1, transformation="tanh")

        # Intermediate LSTM layers
        self.n_layers = n_layers
        if typeNN == 'LSTM':
            self.LayerList = nn.ModuleList([
                LSTMLayer(layer_width, input_dim+1) for _ in range(self.n_layers)
            ])
        elif typeNN == 'Dense':
            self.LayerList = nn.ModuleList([
                DenseLayer(layer_width, layer_width, transformation="tanh") for _ in range(self.n_layers)
            ])
        else:
            raise Exception('typeNN ' + typeNN + ' not supported. Choose one of Dense or LSTM')
        self.typeNN = typeNN
        # Final layer as fully connected with multiple outputs (function values)
        self.final_layer = DenseLayer(num_classes, layer_width, transformation=final_trans)

    def forward(self, t, x):
        '''
        Args:
            t: sampled time inputs
            x: sampled space inputs

        Run the DGM model and obtain fitted function value at the inputs (t,x)
        '''
        # Define input vector as time-space pairs
        X = torch.cat([t, x], 1)
        X = MinMaxScaler().fit_transform(X)
        # Call initial layer
        S = self.initial_layer(X)

        # Call intermediate LSTM layers
        for layer in self.LayerList:
            if self.typeNN == 'LSTM':
                S = layer(S, X)
            elif self.typeNN == 'Dense':
                S = layer(S)
        # Call final layer
        result = self.final_layer(S)

        # Get argmax and result
        op = torch.argmax(result, 1)
        op = op.reshape(op.shape[0], 1).float()

        return op, result

class DenseNet(nn.Module):
    def __init__(self, input_dim, layer_width, n_layers, output_dim, model0, final_trans=None):
        '''
        Args:
            layer_width:
            n_layers:    number of intermediate LSTM layers
            input_dim:   spatial dimension of input data (EXCLUDES time dimension)
            num_classes: number of output classes
            final_trans: transformation used in final layer

        Returns: customized PyTorch model object representing DGM neural network
        '''
        super(DenseNet, self).__init__()
        self.is_regression = (output_dim == 1)
        # Initial layer as fully connected
        # NOTE: to account for time inputs we use input_dim+1 as the input dimensionality
        self.initial_layer = model0

        # Intermediate LSTM layers
        self.n_layers = n_layers
        self.LayerList = nn.ModuleList([DenseLayer( layer_width,input_dim, transformation="tanh")] + [
            DenseLayer(layer_width, layer_width, transformation="tanh") for _ in range(self.n_layers-1)
        ])
        # Final layer as fully connected with multiple outputs (function values)
        self.final_layer = DenseLayer(output_dim, layer_width, transformation=final_trans)

    def forward(self, t, x):
        '''
        Args:
            t: sampled time inputs
            x: sampled space inputs

        Run the DGM model and obtain fitted function value at the inputs (t,x)
        '''
        # Define input vector as time-space pairs

        # Call initial layer
        S = self.initial_layer(t, x)

        # Call intermediate LSTM layers
        for layer in self.LayerList:
            S = layer(S)
        # Call final layer
        result = self.final_layer(S)
        if self.is_regression: return result
        # Get argmax and result
        op = torch.argmax(result, 1)
        op = op.reshape(op.shape[0], 1).float()

        return op, result