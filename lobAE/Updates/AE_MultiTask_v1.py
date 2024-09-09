import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from utils_v1 import set_seed, read_data, prepare_data, split_dataset, CustomDataset, batch_standardize, evaluate_on_test
from AE_simple_v1 import Encoder, Decoder

## Model definition
# Inherit from Decoder in AE_simple.py
class MultiTaskDecoder(Decoder):
    def __init__(self, state_dim, action_dim, hidden_dim, num_layers, output_dims):
        super(MultiTaskDecoder, self).__init__(state_dim, action_dim, hidden_dim, hidden_dim, num_layers)
        
        # Separate output layers for price and volume
        self.price_decoder = nn.Linear(hidden_dim, output_dims['price'])
        self.volume_decoder = nn.Linear(hidden_dim, output_dims['volume'])

    def forward(self, state, action):
        # Use the forward method from the Decoder class for the shared layers
        x = super(MultiTaskDecoder, self).forward(state, action)
        
        # Separate predictions for price and volume
        price_pred = self.price_decoder(x)
        volume_pred = self.volume_decoder(x)
        
        # Combine predictions
        return torch.cat((price_pred, volume_pred), dim=1)

# Define the MultiTaskAutoencoder
class MultiTaskAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, state_dim, action_dim, output_dims, num_layers):
        super(MultiTaskAutoencoder, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, state_dim, num_layers)
        self.decoder = MultiTaskDecoder(state_dim, action_dim, hidden_dim, num_layers, output_dims)

    def forward(self, observation, action, prev_state, prev_action):
        state = self.encoder(observation, prev_state, prev_action)
        prediction = self.decoder(state, action)
        return prediction, state


## Model Training
def train_autoencoder_with_validation(autoencoder, train_loader, val_loader, epochs, learning_rate, batch_size, state_dim, action_dim):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        autoencoder.train()
        total_train_loss = 0
        prev_state = torch.zeros((batch_size, state_dim))  # initialize prev_state as 0
        prev_action = torch.zeros((batch_size, action_dim))  # initialize prev_action as 0

        for obs, action, next_obs, _, _ in train_loader:
            optimizer.zero_grad()
            prediction, state = autoencoder(obs, action, prev_state, prev_action)

            # Separate loss for price and volume
            loss_price = criterion(prediction[:, :4], next_obs[:, :4])
            loss_volume = criterion(prediction[:, 4:], next_obs[:, 4:])
            loss = loss_price + loss_volume
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

            prev_state = state.detach()  # update prev_state with the current state
            prev_action = action.detach()  # update prev_actionwith the current action
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        autoencoder.eval()
        total_val_loss = 0
        with torch.no_grad():
            prev_state = torch.zeros((batch_size, state_dim))
            prev_action = torch.zeros((batch_size, action_dim))

            for obs, action, next_obs, _, _ in val_loader:
                prediction, state = autoencoder(obs, action, prev_state, prev_action)
                # Separate loss for price and volume
                loss_price = criterion(prediction[:, :4], next_obs[:, :4])
                loss_volume = criterion(prediction[:, 4:], next_obs[:, 4:])
                loss = loss_price + loss_volume
                total_val_loss += loss.item()

                prev_state = state.detach()
                prev_action = action.detach()
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f'Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
    
    plt.figure(figsize=(6,4))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

## Main Function
def main():
    set_seed(42)
    data = read_data('nmdp_rl/data/dataset_with_percentile/AAPL_2019-01-02_dataset_2ls.csv')
    observations, actions, next_observations = prepare_data(data)
    obs_train, obs_val, obs_test, act_train, act_val, act_test, next_obs_train, next_obs_val, next_obs_test = split_dataset(
        observations, actions, next_observations
    )

    # Hyperparameters
    state_dim = 32
    action_dim = actions.shape[1]
    input_dim = observations.shape[1] + state_dim + action_dim
    hidden_dim = input_dim
    output_dims = {'price': 4, 'volume': 4}  # Output dimensions for price and volume
    num_layers = 8  # Adjust the depth of the network
    batch_size = 32
    epochs = 50
    learning_rate = 1e-4

    # Prepare DataLoader
    train_dataset = CustomDataset(obs_train, act_train, next_obs_train)
    val_dataset = CustomDataset(obs_val, act_val, next_obs_val)
    test_dataset = CustomDataset(obs_test, act_test, next_obs_test)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=False, collate_fn=batch_standardize, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False, collate_fn=batch_standardize, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False, collate_fn=batch_standardize, drop_last=True)

    # Instantiate and train the model
    autoencoder = MultiTaskAutoencoder(input_dim, hidden_dim, state_dim, action_dim, output_dims, num_layers)
    train_autoencoder_with_validation(autoencoder, train_loader, val_loader, epochs, learning_rate, batch_size, state_dim, action_dim)
    evaluate_on_test(autoencoder, test_loader, batch_size, state_dim, action_dim)

if __name__ == "__main__":
    main()
