import torch
import torch.nn as nn
from utils import MinMaxScaler

def get_activation_function(activation_name):
    """Get activation function by name.

    Args:
        activation_name (str): Name of the activation function.
            One of: "tanh", "relu", "sigmoid" or None.

    Returns:
        function: The corresponding PyTorch activation function.
    """
    activation_map = {
        "tanh": torch.tanh,
        "relu": torch.relu,
        "sigmoid": torch.sigmoid,
        "leaky_relu": nn.functional.leaky_relu,
        "gelu": nn.functional.gelu,
        "selu": nn.functional.selu,
        "softmax": lambda x: nn.functional.softmax(x, dim=-1),
        "none": None
    }
    return activation_map.get(activation_name.lower() if isinstance(activation_name, str) else activation_name, None)


class DenseLayer(nn.Module):
    def __init__(self, output_dim, input_dim, activation="tanh"):
        '''
        Args:
            input_dim:   dimensionality of input data
            output_dim:  number of outputs for dense layer
            activation:  activation function used inside the layer; using
                         None or "none" is equivalent to the identity map

        Returns: customized PyTorch (fully connected) layer object
        '''
        super(DenseLayer, self).__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim

        # Use PyTorch's built-in Linear layer
        self.linear = nn.Linear(input_dim, output_dim)

        # Set activation function
        self.activation = get_activation_function(activation)

    def forward(self, X):
        '''Compute output of a dense layer for a given input X

        Args:
            X: input to layer
        '''
        # Compute dense layer output using PyTorch's linear layer
        S = self.linear(X)

        if self.activation:
            S = self.activation(S)

        return S

## backwd compatibility
class BaseLayer(nn.Module):
    """Base class for neural network layers."""

    def _initialize_weights(self, weights, biases=None):
        """Initialize weights and biases.

        Args:
            weights (list): List of weight parameters.
            biases (list, optional): List of bias parameters.
        """
        # Xavier initialization for weights
        for param in weights:
            nn.init.xavier_normal_(param)

        # Initialize biases to zero
        if biases:
            for param in biases:
                nn.init.zeros_(param)

class LSTMLayer(BaseLayer):
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

        # Get activation functions
        self.trans1 = get_activation_function(trans1) or torch.tanh
        self.trans2 = get_activation_function(trans2) or torch.tanh

        # LSTM layer parameters
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
        weights = [self.Uz, self.Ug, self.Ur, self.Uh, self.Wz, self.Wg, self.Wr, self.Wh]
        biases = [self.bz, self.bg, self.br, self.bh]
        self._initialize_weights(weights, biases)

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
##

class BaseNet(nn.Module):
    """Base class for neural network models."""

    def __init__(self):
        super(BaseNet, self).__init__()

    def preprocess_input(self, t, x):
        """Preprocess input data.

        Args:
            t: time inputs
            x: space inputs

        Returns:
            torch.Tensor: Preprocessed input data
        """
        # Concatenate time and space inputs
        X = torch.cat([t, x], 1)
        # Scale inputs
        return MinMaxScaler().fit_transform(X)


class TransformerEncoder(nn.Module):
    """Custom Transformer Encoder for time series data."""

    def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward, activation="relu", dropout=0.1):
        """
        Args:
            input_dim: Input feature dimension
            d_model: Model dimension for transformer
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Feed-forward network dimension
            activation: Activation function to use
            dropout: Dropout rate
        """
        super(TransformerEncoder, self).__init__()

        # Initial projection to model dimension
        self.input_projection = nn.Linear(input_dim, d_model)

        # Position encoding (learned)
        self.pos_encoder = nn.Parameter(torch.empty(1, 1, d_model))
        nn.init.normal_(self.pos_encoder, mean=0, std=0.02)

        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True
        )

        # Stack encoder layers
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection
        self.output_projection = nn.Linear(d_model, d_model)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim]

        Returns:
            Output tensor of shape [batch_size, d_model]
        """
        # Project input to model dimension
        x = self.input_projection(x)

        # Add positional encoding
        x = x + self.pos_encoder

        # Apply transformer encoder
        x = self.transformer_encoder(x)

        # Get the final token representation
        x = x[:, -1, :]  # Take the last token for sequence representation

        # Project to output dimension
        x = self.output_projection(x)

        return x


class NeuralNet(BaseNet):
    def __init__(self, layer_width, n_layers, input_dim, output_dim=1, final_activation=None,
                 typeNN='LSTM', is_regression=True, hidden_activation="tanh",
                 transformer_config=None):
        '''
        Args:
            layer_width:        width of intermediate layers
            n_layers:           number of intermediate layers
            input_dim:          spatial dimension of input data (EXCLUDES time dimension)
            output_dim:         dimensionality of output (1 for regression, >1 for classification)
            final_activation:   activation function used in final layer
            typeNN:             type of network ('LSTM', 'Dense', or 'Transformer')
            is_regression:      whether this is a regression model (True) or classification (False)
            hidden_activation:  activation function used in hidden layers
            transformer_config: configuration for transformer model (if typeNN='Transformer')
                                dict with keys: nhead, dim_feedforward, dropout

        Returns: customized PyTorch model object representing a neural network
        '''
        super(NeuralNet, self).__init__()

        self.is_regression = is_regression
        self.typeNN = typeNN

        # Calculate input size (add 1 for time dimension)
        input_size = input_dim + 1

        # Initial layer as fully connected
        self.initial_layer = DenseLayer(layer_width, input_size, activation=hidden_activation)

        # Default transformer config
        default_transformer_config = {
            'nhead': 5,  # Number of attention heads
            'dim_feedforward': 2048,  # Dimension of feedforward network
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

        # Intermediate layers
        if typeNN == 'LSTM':
            # Use PyTorch's built-in LSTM layer
            # self.lstm = nn.LSTM(
            #     input_size=input_size + layer_width,  # Concatenated input and previous layer output
            #     hidden_size=layer_width,
            #     num_layers=n_layers,
            #     batch_first=True
            # )
            self.LayerList = nn.ModuleList([
                LSTMLayer(layer_width, input_dim+1) for _ in range(n_layers)
            ])
        elif typeNN == 'Dense':
            self.layers = nn.ModuleList([
                DenseLayer(layer_width, layer_width, activation=hidden_activation)
                for _ in range(n_layers)
            ])
        elif typeNN == 'Transformer':
            self.transformer = TransformerEncoder(
                input_dim=input_size + layer_width,  # Concatenated input and previous layer output
                d_model=layer_width,
                nhead=transformer_config['nhead'],
                num_layers=n_layers,
                dim_feedforward=transformer_config['dim_feedforward'],
                activation=hidden_activation,
                dropout=transformer_config['dropout']
            )
        else:
            raise ValueError(f'typeNN {typeNN} not supported. Choose one of Dense, LSTM, or Transformer')

        # Final layer as fully connected
        self.final_layer = DenseLayer(output_dim, layer_width, activation=final_activation)

    def forward(self, t, x):
        '''
        Args:
            t: sampled time inputs
            x: sampled space inputs

        Run the model and obtain predictions at the inputs (t,x)
        '''
        # Preprocess input
        X = self.preprocess_input(t, x)

        # Call initial layer
        S = self.initial_layer(X)

        # Process through intermediate layers
        if self.typeNN == 'LSTM':
            # # For LSTM, we need to reshape and concatenate the inputs
            # batch_size = X.size(0)
            #
            # # Concatenate the current state S with input X
            # combined = torch.cat([S, X], dim=1)
            #
            # # Reshape for LSTM: (batch, seq_len=1, features)
            # combined = combined.unsqueeze(1)
            #
            # # Process through LSTM
            # output, _ = self.lstm(combined)
            #
            # # Extract the output of the last time step
            # S = output[:, -1, :]
            for layer in self.layers:
                S = layer(S, X)
        elif self.typeNN == 'Dense':
            for layer in self.layers:
                S = layer(S)

        elif self.typeNN == 'Transformer':
            # For Transformer, reshape and concatenate similar to LSTM
            combined = torch.cat([S, X], dim=1)
            combined = combined.unsqueeze(1)  # Add sequence dimension (batch, seq_len=1, features)

            # Process through Transformer
            S = self.transformer(combined)

        # Call final layer
        result = self.final_layer(S)

        # For regression, just return the result
        if self.is_regression:
            return result

        # For classification, return predicted class and probabilities
        op = torch.argmax(result, 1)
        op = op.reshape(op.shape[0], 1).float()
        return op, result


# For backward compatibility
class DGMNet(NeuralNet):
    def __init__(self, layer_width, n_layers, input_dim, output_dim=1, final_trans=None,
                 typeNN='LSTM', hidden_activation="tanh", transformer_config=None):
        '''Regression-focused neural network model

        Args: Same as NeuralNet with backward compatibility
        '''
        super(DGMNet, self).__init__(
            layer_width=layer_width,
            n_layers=n_layers,
            input_dim=input_dim,
            output_dim=output_dim,
            final_activation=final_trans,  # Renamed parameter
            typeNN=typeNN,
            is_regression=True,
            hidden_activation=hidden_activation,
            transformer_config=transformer_config
        )


class PIANet(NeuralNet):
    def __init__(self, layer_width, n_layers, input_dim, num_classes, final_trans=None,
                 typeNN='LSTM', hidden_activation="tanh", transformer_config=None):
        '''Classification-focused neural network model

        Args: Same as NeuralNet, with num_classes instead of output_dim
        '''
        super(PIANet, self).__init__(
            layer_width=layer_width,
            n_layers=n_layers,
            input_dim=input_dim,
            output_dim=num_classes,
            final_activation=final_trans,  # Renamed parameter
            typeNN=typeNN,
            is_regression=False,
            hidden_activation=hidden_activation,
            transformer_config=transformer_config
        )


class DenseNet(BaseNet):
    def __init__(self, input_dim, layer_width, n_layers, output_dim, model0,
                 final_activation=None, hidden_activation="tanh"):
        '''
        Args:
            input_dim:         spatial dimension of input data
            layer_width:       width of intermediate layers
            n_layers:          number of intermediate layers
            output_dim:        dimensionality of output
            model0:            initial model to use
            final_activation:  activation function used in final layer
            hidden_activation: activation function for hidden layers

        Returns: customized PyTorch model using another model's output as input
        '''
        super(DenseNet, self).__init__()

        self.is_regression = (output_dim == 1)

        # Initial layer uses output from another model
        self.initial_layer = model0

        # Use PyTorch's Sequential for cleaner implementation of dense layers
        layers = [DenseLayer(layer_width, input_dim, activation=hidden_activation)]
        for _ in range(n_layers - 1):
            layers.append(DenseLayer(layer_width, layer_width, activation=hidden_activation))
        self.layers = nn.Sequential(*layers)

        # Final layer
        self.final_layer = DenseLayer(output_dim, layer_width, activation=final_activation)

    def forward(self, t, x):
        '''
        Args:
            t: sampled time inputs
            x: sampled space inputs

        Run the model and obtain predictions
        '''
        # Call initial layer (which is another model)
        S = self.initial_layer(t, x)

        # Call intermediate layers
        S = self.layers(S)

        # Call final layer
        result = self.final_layer(S)

        if self.is_regression:
            return result

        # For classification, get argmax and result
        op = torch.argmax(result, 1)
        op = op.reshape(op.shape[0], 1).float()

        return op, result