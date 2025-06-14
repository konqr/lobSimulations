import os
os.environ["PYTORCH_ENABLE_MEM_EFFICIENT_SDPA"] = "0"
import torch
import torch.nn as nn
from HJBQVI.utils import MinMaxScaler

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
        torch.nn.init.xavier_normal_(self.linear.weight)
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
            batch_first=True,
            norm_first=True
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
        x = self.transformer_encoder(x, mask=None)

        # Get the final token representation
        x = x[:, -1, :]  # Take the last token for sequence representation

        # Project to output dimension
        x = self.output_projection(x)

        return x

# Replace your TransformerEncoder with this more manual implementation
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
            #Use PyTorch's built-in LSTM layer
            self.lstm = nn.LSTM(
                input_size=input_size + layer_width,  # Concatenated input and previous layer output
                hidden_size=layer_width,
                num_layers=n_layers,
                batch_first=True
            )
        elif typeNN == 'Dense':
            self.layers = nn.ModuleList([
                DenseLayer(layer_width, layer_width, activation=hidden_activation)
                for _ in range(n_layers)
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
            # For LSTM, we need to reshape and concatenate the inputs
            batch_size = X.size(0)

            # Concatenate the current state S with input X
            combined = torch.cat([S, X], dim=1)

            # Reshape for LSTM: (batch, seq_len=1, features)
            combined = combined.unsqueeze(1)
            with torch.backends.cudnn.flags(enabled=False):
                # Process through LSTM
                output, _ = self.lstm(combined)

            # Extract the output of the last time step
            S = output[:, -1, :]
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

class MLP(BaseNet):
    def __init__(self, input_dim, layer_width, n_layers, output_dim,
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
        super(MLP, self).__init__()

        self.is_regression = (output_dim == 1)
        self.initial_layer = DenseLayer(layer_width, input_dim, activation=hidden_activation)

        # Use PyTorch's Sequential for cleaner implementation of dense layers
        layers = [DenseLayer(layer_width, layer_width, activation=hidden_activation)]
        for _ in range(n_layers - 1):
            layers.append(DenseLayer(layer_width, layer_width, activation=hidden_activation))
        self.layers = nn.Sequential(*layers)

        # Final layer
        self.final_layer = DenseLayer(output_dim, layer_width, activation=final_activation)
        self.final_activation = final_activation
    def forward(self, x):
        '''
        Args:
            x: sampled space inputs

        Run the model and obtain predictions
        '''
        # Call initial layer (which is another model)
        # x = MinMaxScaler().fit_transform(x)
        S = self.initial_layer(x)

        # Call intermediate layers
        S = self.layers(S)

        # Call final layer
        if self.final_activation != 'softmax':
            result = self.final_layer(S)
        else:
            # print(S - torch.max(S))
            result = self.final_layer(S) # - torch.max(S))

        if self.is_regression:
            return result

        # For classification, get argmax and result
        op = torch.argmax(result.clone(), 1)
        op = op.reshape(op.shape[0], 1).float()

        return op, result

class ActorCriticMLP(BaseNet):
    def __init__(self, input_dim, layer_width, n_layers,
                 actor_output_dim, actor_activation="softmax",
                 hidden_activation="tanh", q_function = True):
        '''
        Args:
            input_dim:         spatial dimension of input data
            layer_width:       width of intermediate layers
            n_layers:          number of intermediate layers
            actor_output_dim:  dimensionality of actor output (action space)
            actor_activation:  activation function used in actor's output layer
            hidden_activation: activation function for hidden layers

        Returns: A shared MLP network with separate heads for actor and critic
        '''
        super(ActorCriticMLP, self).__init__()

        # Shared feature extraction layers
        self.initial_layer = DenseLayer(layer_width, input_dim, activation=hidden_activation)

        # Build shared hidden layers
        layers = []
        for _ in range(n_layers):
            layers.append(DenseLayer(layer_width, layer_width, activation=hidden_activation))
        self.shared_layers = nn.Sequential(*layers)

        # Actor output layer (policy network)
        self.actor_output = DenseLayer(actor_output_dim, layer_width, activation=actor_activation)

        # Critic output layer (value function) - always outputs a single value
        opDim = actor_output_dim if q_function else 1
        self.critic_output = DenseLayer(opDim, layer_width, activation=None)

        self.actor_activation = actor_activation

    def forward(self, x):
        '''
        Process inputs through the shared network, then through
        separate actor and critic heads

        Args:
            x: input state

        Returns:
            actor_output: policy distribution or action
            critic_output: value estimate
        '''
        # Forward through shared layers
        features = self.initial_layer(x)
        features = self.shared_layers(features)

        # Actor head
        actor_output = self.actor_output(features)

        # Critic head (value function)
        critic_output = self.critic_output(features)

        # For classification policy, get argmax action
        if self.actor_activation == 'softmax':
            action = torch.argmax(actor_output.clone(), 1)
            action = action.reshape(action.shape[0], 1).float()
            return action, actor_output, critic_output

        return actor_output, critic_output

class ActorCriticSGMLP(nn.Module):
    def __init__(self, input_dim, layer_width, n_layers,
                 actor_output_dim, actor_log_std_min=-20, actor_log_std_max=2,
                 hidden_activation="tanh", q_function=True):
        '''
        Args:
            input_dim:         spatial dimension of input data
            layer_width:       width of intermediate layers
            n_layers:          number of intermediate layers
            actor_output_dim:  dimensionality of actor output (action space)
            actor_log_std_min: minimum value for log standard deviation
            actor_log_std_max: maximum value for log standard deviation
            hidden_activation: activation function for hidden layers
            q_function:        whether to use Q-function or V-function for critic

        Returns: A shared MLP network with separate heads for actor and critic
        '''
        super(ActorCriticSGMLP, self).__init__()

        # Shared feature extraction layers
        self.initial_layer = DenseLayer(layer_width, input_dim, activation=hidden_activation)

        # Build shared hidden layers
        layers = []
        for _ in range(n_layers):
            layers.append(DenseLayer(layer_width, layer_width, activation=hidden_activation))
        self.shared_layers = nn.Sequential(*layers)

        # Actor mean and log std layers for Gaussian policy
        self.actor_mean = DenseLayer(actor_output_dim, layer_width, activation=None)
        self.actor_log_std = DenseLayer(actor_output_dim, layer_width, activation=None)

        # Parameters for log std clamping
        self.actor_log_std_min = actor_log_std_min
        self.actor_log_std_max = actor_log_std_max

        # Critic output layer
        opDim = actor_output_dim if q_function else 1
        self.critic_output = DenseLayer(opDim, layer_width, activation=None)

    def forward(self, x):
        '''
        Process inputs through the shared network, then through
        separate actor and critic heads

        Args:
            x: input state

        Returns:
            action: sampled action after squashing
            actor_output: mean and log std of policy distribution
            critic_output: value estimate
        '''
        # Forward through shared layers
        features = self.initial_layer(x)
        features = self.shared_layers(features)

        # Actor mean and log std
        actor_mean = self.actor_mean(features)
        actor_log_std = self.actor_log_std(features)

        # Clamp log std to prevent extreme values
        actor_log_std = torch.clamp(
            actor_log_std,
            min=self.actor_log_std_min,
            max=self.actor_log_std_max
        )

        # Create distribution
        actor_std = torch.exp(actor_log_std)
        actor_dist = torch.distributions.Normal(actor_mean, actor_std)

        # Sample action and apply tanh squashing
        raw_action = actor_dist.rsample()
        action = torch.tanh(raw_action)

        # Compute log probability with squashing correction
        log_prob = actor_dist.log_prob(raw_action)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        # Critic head (value function)
        critic_output = self.critic_output(features)

        return action, (actor_mean, actor_log_std, log_prob), critic_output

    def get_log_prob(self, x, actions):
        '''
        Compute log probability of given actions

        Args:
            x: input state
            actions: input actions to compute log probability for

        Returns:
            log_prob: log probability of actions
        '''
        # Forward through shared layers
        features = self.initial_layer(x)
        features = self.shared_layers(features)

        # Actor mean and log std
        actor_mean = self.actor_mean(features)
        actor_log_std = self.actor_log_std(features)

        # Clamp log std to prevent extreme values
        actor_log_std = torch.clamp(
            actor_log_std,
            min=self.actor_log_std_min,
            max=self.actor_log_std_max
        )

        # Create distribution
        actor_std = torch.exp(actor_log_std)
        actor_dist = torch.distributions.Normal(actor_mean, actor_std)

        # Inverse squashing
        raw_actions = torch.arctanh(torch.clamp(actions, -1+1e-6, 1-1e-6))

        # Compute log probability with squashing correction
        log_prob = actor_dist.log_prob(raw_actions)
        log_prob -= torch.log(1 - actions.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return log_prob

class ResidualBlock(nn.Module):
    def __init__(self, width, activation="tanh"):
        '''
        Residual block with skip connection

        Args:
            width: width of the block (input and output dimensions)
            activation: activation function for hidden layers
        '''
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            DenseLayer(width, width, activation=activation),
            DenseLayer(width, width, activation=activation)
        )

    def forward(self, x):
        '''Forward pass with residual connection'''
        return x + self.block(x)


class ActorResNet(BaseNet):
    def __init__(self, input_dim, layer_width, n_layers,
                 output_dim, output_activation="softmax",
                 hidden_activation="tanh"):
        '''
        Actor network with ResNet architecture for policy learning

        Args:
            input_dim:         spatial dimension of input data
            layer_width:       width of intermediate layers
            n_layers:          number of residual blocks
            output_dim:        dimensionality of actor output (action space)
            output_activation: activation function used in output layer
            hidden_activation: activation function for hidden layers
        '''
        super(ActorResNet, self).__init__()

        # Initial layer to project input to layer_width
        self.initial_layer = DenseLayer(layer_width, input_dim, activation=hidden_activation)

        # Build residual blocks
        blocks = []
        for _ in range(n_layers):
            blocks.append(ResidualBlock(layer_width, activation=hidden_activation))
        self.residual_blocks = nn.Sequential(*blocks)

        # Output layer (policy network)
        self.output_layer = DenseLayer(output_dim, layer_width, activation=output_activation)

    def forward(self, x):
        '''
        Forward pass through actor ResNet

        Args:
            x: input state

        Returns:
            policy distribution or action probabilities
        '''
        x = self.initial_layer(x)
        x = self.residual_blocks(x)
        output = self.output_layer(x)
        return output

class ActorMLP(BaseNet):
    def __init__(self, input_dim, layer_width, n_layers,
                 output_dim, output_activation="softmax",
                 hidden_activation="tanh"):
        '''
        Actor network for policy learning

        Args:
            input_dim:         spatial dimension of input data
            layer_width:       width of intermediate layers
            n_layers:          number of intermediate layers
            output_dim:        dimensionality of actor output (action space)
            output_activation: activation function used in output layer
            hidden_activation: activation function for hidden layers
        '''
        super(ActorMLP, self).__init__()

        # Initial layer
        self.initial_layer = DenseLayer(layer_width, input_dim, activation=hidden_activation)

        # Build hidden layers
        layers = []
        for _ in range(n_layers):
            layers.append(DenseLayer(layer_width, layer_width, activation=hidden_activation))
        self.hidden_layers = nn.Sequential(*layers)

        # Output layer (policy network)
        self.output_layer = DenseLayer(output_dim, layer_width, activation=output_activation)

    def forward(self, x):
        '''
        Forward pass through actor network

        Args:
            x: input state

        Returns:
            policy distribution or action probabilities
        '''
        x = self.initial_layer(x)
        x = self.hidden_layers(x)
        output = self.output_layer(x)
        return output


class CriticMLP(BaseNet):
    def __init__(self, input_dim, layer_width, n_layers,
                 output_dim=1, hidden_activation="tanh", q_function=False):
        '''
        Critic network for value function estimation

        Args:
            input_dim:         spatial dimension of input data
            layer_width:       width of intermediate layers
            n_layers:          number of intermediate layers
            output_dim:        dimensionality of critic output (1 for V, action_dim for Q)
            hidden_activation: activation function for hidden layers
            q_function:        whether this is a Q-function (True) or V-function (False)
        '''
        super(CriticMLP, self).__init__()

        # Initial layer
        self.initial_layer = DenseLayer(layer_width, input_dim, activation=hidden_activation)

        # Build hidden layers
        layers = []
        for _ in range(n_layers):
            layers.append(DenseLayer(layer_width, layer_width, activation=hidden_activation))
        self.hidden_layers = nn.Sequential(*layers)

        # Output layer (value function) - no activation for value estimation
        self.output_layer = DenseLayer(output_dim, layer_width, activation=None)

        self.q_function = q_function

    def forward(self, x):
        '''
        Forward pass through critic network

        Args:
            x: input state

        Returns:
            value estimate
        '''
        x = self.initial_layer(x)
        x = self.hidden_layers(x)
        output = self.output_layer(x)
        return output


class ActorCriticSeparate(nn.Module):
    def __init__(self, input_dim, layer_width, n_layers,
                 actor_output_dim, actor_activation="softmax",
                 hidden_activation="tanh", q_function=True):
        '''
        Separate Actor-Critic networks

        Args:
            input_dim:         spatial dimension of input data
            layer_width:       width of intermediate layers
            n_layers:          number of intermediate layers
            actor_output_dim:  dimensionality of actor output (action space)
            actor_activation:  activation function used in actor's output layer
            hidden_activation: activation function for hidden layers
            q_function:        whether critic is Q-function (True) or V-function (False)
        '''
        super(ActorCriticSeparate, self).__init__()
        # Create separate actor and critic networks
        self.actor = ActorMLP(
            input_dim=input_dim,
            layer_width=layer_width,
            n_layers=n_layers,
            output_dim=actor_output_dim,
            output_activation=actor_activation,
            hidden_activation=hidden_activation
        )

        critic_output_dim = actor_output_dim if q_function else 1
        self.critic = CriticMLP(
            input_dim=input_dim,
            layer_width=layer_width,
            n_layers=n_layers,
            output_dim=critic_output_dim,
            hidden_activation=hidden_activation,
            q_function=q_function
        )

    def forward(self, x):
        '''
        Process inputs through both networks

        Args:
            x: input state

        Returns:
            actor_output: policy distribution or action
            critic_output: value estimate
        '''
        actor_output = self.actor(x)
        critic_output = self.critic(x)
        return actor_output, critic_output

    def actor_forward(self, x):
        '''Forward pass through actor only'''
        return self.actor(x)

    def critic_forward(self, x):
        '''Forward pass through critic only'''
        return self.critic(x)

    def parameters(self):
        '''Get all parameters from both networks'''
        import itertools
        return itertools.chain(self.actor.parameters(), self.critic.parameters())

    def actor_parameters(self):
        '''Get actor parameters only'''
        return self.actor.parameters()

    def critic_parameters(self):
        '''Get critic parameters only'''
        return self.critic.parameters()