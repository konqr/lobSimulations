import torch.nn as nn
import torch.nn.functional as F

from AE_MultiTask_v1 import MultiTaskDecoder, main
from AE_MT_Attention_v1 import Attention, EncoderWithAttention

## Model definition
class EncoderWithAttentionAndCNN(EncoderWithAttention):
    def __init__(self, input_dim, hidden_dim, state_dim, num_layers, attention_dim, conv_filters, kernel_size):
        # Initialize the parent class (EncoderWithAttention)
        super(EncoderWithAttentionAndCNN, self).__init__(input_dim, hidden_dim, state_dim, num_layers, attention_dim)
        # Add 1D Convolutional layer
        self.conv1d = nn.Conv1d(in_channels=input_dim, out_channels=conv_filters, kernel_size=kernel_size, padding=kernel_size//2)
        # Override attention layer to match the output of conv1d
        self.attention = Attention(conv_filters, attention_dim)
        # Override hidden layers
        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(nn.Linear(conv_filters, hidden_dim))
        for _ in range(num_layers - 2):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
        if num_layers > 1:
            self.hidden_layers.append(nn.Linear(hidden_dim, state_dim))
        else:
            self.hidden_layers[0] = nn.Linear(conv_filters, state_dim)
    
    def forward(self, x):
        if x.dim() == 2:  # Ensure x is 3D: (batch_size, seq_len, feature_dim)
            x = x.unsqueeze(1)
        # Apply 1D Convolution
        x = x.permute(0, 2, 1)  # Change dimension to (batch_size, feature_dim, seq_len)
        x = F.relu(self.conv1d(x))
        x = x.permute(0, 2, 1)  # Change back to (batch_size, seq_len, conv_filters)
        # Apply attention mechanism (inherited from EncoderWithAttention)
        x, _ = self.attention(x)
        # Pass through the hidden layers (inherited from EncoderWithAttention)
        for layer in self.hidden_layers:
            x = F.leaky_relu(layer(x))
        return x.squeeze(1)


# Define the Autoencoder
class AutoencoderWithAttentionAndCNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, state_dim, action_dim, output_dims, num_layers, attention_dim, conv_filters, kernel_size):
        super(AutoencoderWithAttentionAndCNN, self).__init__()
        self.encoder = EncoderWithAttentionAndCNN(input_dim, hidden_dim, state_dim, num_layers, attention_dim, conv_filters, kernel_size)
        self.decoder = MultiTaskDecoder(state_dim, action_dim, hidden_dim, num_layers, output_dims)

    def forward(self, observation, action):
        state = self.encoder(observation)
        prediction = self.decoder(state, action)
        return prediction


if __name__ == "__main__":
    main()
