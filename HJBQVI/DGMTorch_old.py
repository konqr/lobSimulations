import torch
import torch.nn as nn
from HJBQVI.utils import MinMaxScaler
import torch.nn.functional as F
from torch.distributions import Categorical

class LSTMLayer(nn.Module):
    def __init__(self, output_dim, input_dim, trans1="tanh", trans2="tanh", ln = True):
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

        # layer norm
        self.ln = ln
        if ln:

            self.ln_Z = nn.LayerNorm(output_dim)
            self.ln_G = nn.LayerNorm(output_dim)
            self.ln_R = nn.LayerNorm(output_dim)
            self.ln_H = nn.LayerNorm(output_dim)


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
        if self.ln:
            Z = self.trans1(self.ln_Z(torch.add(torch.matmul(X, self.Uz), torch.matmul(S, self.Wz)) + self.bz))
            G = self.trans1(self.ln_G(torch.add(torch.matmul(X, self.Ug), torch.matmul(S, self.Wg)) + self.bg))
            R = self.trans1(self.ln_R(torch.add(torch.matmul(X, self.Ur), torch.matmul(S, self.Wr)) + self.br))

            H = self.trans2(self.ln_H(torch.add(torch.matmul(X, self.Uh), torch.matmul(torch.mul(S, R), self.Wh)) + self.bh))
        else:
            Z = self.trans1(torch.add(torch.matmul(X, self.Uz), torch.matmul(S, self.Wz)) + self.bz)
            G = self.trans1(torch.add(torch.matmul(X, self.Ug), torch.matmul(S, self.Wg)) + self.bg)
            R = self.trans1(torch.add(torch.matmul(X, self.Ur), torch.matmul(S, self.Wr)) + self.br)

            H = self.trans2(torch.add(torch.matmul(X, self.Uh), torch.matmul(torch.mul(S, R), self.Wh)) + self.bh)

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
                "relu": torch.relu,
                "sigmoid":torch.sigmoid
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

class CustomTransformerEncoder(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward, dropout=0.1, activation='relu'):
        super(CustomTransformerEncoder, self).__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.empty(1, 1, d_model))
        nn.init.normal_(self.pos_encoder, mean=0, std=0.02)

        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d_model),
                nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True),
                nn.Dropout(dropout),
                nn.LayerNorm(d_model),
                nn.Linear(d_model, dim_feedforward),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward, d_model),
                nn.Dropout(dropout)
            ) for _ in range(num_layers)
        ])

        self.output_projection = nn.Linear(d_model, d_model)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.input_projection(x)
        batch_size, seq_len = x.shape[0], x.shape[1]
        pos_encoding = self.pos_encoder.expand(batch_size, seq_len, -1)
        x = x + pos_encoding

        for layer in self.layers:
            norm1, attn, dropout1, norm2, linear1, relu, dropout2, linear2, dropout3 = layer

            # First sublayer (attention)
            residual = x
            x = norm1(x)
            x_attn, _ = attn(x, x, x)
            x = residual + dropout1(x_attn)

            # Second sublayer (feedforward)
            residual = x
            x = norm2(x)
            x = linear1(x)
            x = relu(x)
            x = dropout2(x)
            x = linear2(x)
            x = residual + dropout3(x)

        x = self.final_norm(x)
        x = x[:, -1, :]  # Take the last token
        x = self.output_projection(x)
        return x

class DGMNet(nn.Module):
    def __init__(self, layer_width, n_layers, input_dim, output_dim = 1, final_trans=None, typeNN = 'LSTM', hidden_activation='tanh', transformer_config=None):
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
        input_size = input_dim + 1

        # Initial layer as fully connected
        self.initial_layer = DenseLayer(layer_width, input_size, transformation=hidden_activation)

        # Default transformer config
        default_transformer_config = {
            'nhead': 5,  # Number of attention heads
            'dim_feedforward': 256,  # Dimension of feedforward network
            'dropout': 0.1,  # Dropout rate
        }

        # Use provided transformer config or default
        if transformer_config is None:
            transformer_config = default_transformer_config
        else:
            # Update default with provided config
            for key, value in default_transformer_config.items():
                if key not in transformer_config:
                    transformer_config[key] = value

        # Intermediate LSTM layers
        self.n_layers = n_layers
        self.final_trans = final_trans
        if typeNN == 'LSTM':
            self.LayerList = nn.ModuleList([
                LSTMLayer(layer_width, input_dim+1, trans1=hidden_activation, trans2='tanh', ln= False) for _ in range(self.n_layers)
            ])
        elif typeNN == 'Dense':
            self.LayerList = nn.ModuleList([
                DenseLayer(layer_width, layer_width, transformation=hidden_activation) for _ in range(self.n_layers)
            ])
        elif typeNN == 'Transformer':
            self.transformer = CustomTransformerEncoder(
                input_dim=input_size + layer_width,  # Concatenated input and previous layer output
                d_model=layer_width,
                nhead=transformer_config['nhead'],
                num_layers=n_layers,
                dim_feedforward=transformer_config['dim_feedforward'],
                activation=hidden_activation,
                dropout=transformer_config['dropout']
            )
        else:
            raise Exception('typeNN ' + typeNN + ' not supported. Choose one of Dense or LSTM')
        self.typeNN = typeNN
        # Final layer as fully connected with a single output (function value)
        self.final_layer = DenseLayer(output_dim, layer_width, transformation=final_trans)
        if self.final_trans is None:
            self.scale = nn.Parameter(torch.tensor(1.0))
            self.shift = nn.Parameter(torch.tensor(0.0))
    def forward(self, t, x):
        '''
        Args:
            t: sampled time inputs
            x: sampled space inputs

        Run the DGM model and obtain fitted function value at the inputs (t,x)
        '''
        # Define input vector as time-space pairs

        X = torch.cat([t, x], 1)
        # X = MinMaxScaler().fit_transform(X)
        # Call initial layer
        S = self.initial_layer(X)


        if self.typeNN == 'Transformer':
            # For Transformer, reshape and concatenate similar to LSTM
            combined = torch.cat([S, X], dim=1)
            combined = combined.unsqueeze(1)  # Add sequence dimension (batch, seq_len=1, features)

            # Process through Transformer
            S = self.transformer(combined)
        else:
            # Call intermediate LSTM layers
            for layer in self.LayerList:
                if self.typeNN == 'LSTM':
                    S = layer(S, X)
                elif self.typeNN == 'Dense':
                    S = layer(S)

        # Call final layer
        if self.final_trans is None:
            result = self.scale*self.final_layer(S) + self.shift
        else:
            result = self.final_layer(S)

        return result


class PIANet(DGMNet):
    def __init__(self, layer_width, n_layers, input_dim, num_classes, final_trans=None,
                 typeNN='LSTM', hidden_activation='tanh', transformer_config=None):
        super(PIANet, self).__init__(
            layer_width=layer_width,
            n_layers=n_layers,
            input_dim=input_dim,
            output_dim=num_classes,
            typeNN=typeNN,
            hidden_activation=hidden_activation,
            final_trans='sigmoid',
            transformer_config=transformer_config
        )

    def forward(self, t, x, stochastic=False, tau=1.0, hard=False, max_prob=0.75):
        logits = super(PIANet, self).forward(t, x)  # [batch_size, num_classes]

        if stochastic:
            # # --- Clamp logits to enforce max probability ---
            # sum_exp = torch.exp(logits).sum(dim=1)
            # clamp_val = torch.log(max_prob*sum_exp)
            # logits = torch.clamp(logits, max = clamp_val)

            # Recompute Gumbel-Softmax
            probs = F.gumbel_softmax(logits, tau=tau, hard=hard, dim=1)
            # Sample categorical
            dist = Categorical(probs)
            actions = dist.sample().unsqueeze(1).float()
        else:
            actions = torch.argmax(logits, dim=1, keepdim=True).float()

        return actions, logits


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