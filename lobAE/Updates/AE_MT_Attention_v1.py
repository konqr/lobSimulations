import torch
import torch.nn as nn
import torch.nn.functional as F

from AE_simple_v1 import Encoder
from AE_MultiTask_v1 import MultiTaskDecoder, main

## Model definition
# Define the Attention Mechanism
class Attention(nn.Module):
    def __init__(self, input_dim, attention_dim):
        super(Attention, self).__init__()
        self.query_layer = nn.Linear(input_dim, attention_dim)
        self.key_layer = nn.Linear(input_dim, attention_dim)
        self.value_layer = nn.Linear(input_dim, attention_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
         # Ensure x is 3D: (batch_size, seq_len, feature_dim)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add a dimension to make it 3D
        # Query, Key, Value
        query = self.query_layer(x)
        key = self.key_layer(x)
        value = self.value_layer(x)

        # Scaled Dot-Product Attention
        attention_scores = torch.bmm(query, key.transpose(1, 2)) / (key.size(-1) ** 0.5)
        attention_weights = self.softmax(attention_scores)

        # Apply attention weights to the values
        attended_values = torch.bmm(attention_weights, value)
        return attended_values, attention_weights

# Define the Encoder with Attention
class EncoderWithAttention(Encoder):
    def __init__(self, input_dim, hidden_dim, state_dim, num_layers, attention_dim):
        # Initialize the parent class (Encoder)
        super(EncoderWithAttention, self).__init__(input_dim, hidden_dim, state_dim, num_layers)
        # Add attention layer
        self.attention = Attention(input_dim, attention_dim)
    
    def forward(self, observation, prev_state, prev_action):
        # Combine observation, previous state, and previous action
        x = torch.cat((observation, prev_state, prev_action), dim=1)
        # Apply attention mechanism before passing through the hidden layers
        x, _ = self.attention(x)
        # Pass through the hidden layers from the parent class
        for layer in self.hidden_layers:
            x = F.leaky_relu(layer(x))
        return x.squeeze(1)


# Define the Autoencoder
class AutoencoderWithAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, state_dim, action_dim, output_dims, num_layers, attention_dim):
        super(AutoencoderWithAttention, self).__init__()
        self.encoder = EncoderWithAttention(input_dim, hidden_dim, state_dim, num_layers, attention_dim)
        self.decoder = MultiTaskDecoder(state_dim, action_dim, hidden_dim, num_layers, output_dims)

    def forward(self, observation, action, prev_state, prev_action):
        state = self.encoder(observation, prev_state, prev_action)
        prediction = self.decoder(state, action)
        return prediction, state


if __name__ == "__main__":
    main()