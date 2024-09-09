import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from utils_v1 import set_seed, read_data, prepare_data, split_dataset, CustomDataset, batch_standardize, evaluate_on_test

'''
What to do next:
- 1. Add quantile variable to position
- 2. Encoder takes in the previous S, and the previous action as well
(Not really)
- 3. Play with the number of dimensions of S_n, see if the performance increases with increasing dims
(Not really)
- 4. Instead of copying the whole python file in AE_MultiTask.py, just import the functions from AE_simple.py, 
or create a utils.py file with all the common functions
(Finish)
- 5. Loss Function Choices
'''

## Model definition
# Define the Encoder with configurable depth
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, state_dim, num_layers):
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.hidden_layers = nn.ModuleList()

        # Initialize hidden layers
        self.hidden_layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
        if num_layers > 1:
            self.hidden_layers.append(nn.Linear(hidden_dim, state_dim))
        else:
            self.hidden_layers[0] = nn.Linear(input_dim, state_dim)
        # self.relu = nn.ReLU()

    def forward(self, observation, prev_state, prev_action):
        # combine O(n), S(n-1), U(n-1)
        x = torch.cat((observation, prev_state, prev_action), dim=1)
        for layer in self.hidden_layers:
            x = F.leaky_relu(layer(x))
        return x

# Define the Decoder with configurable depth
class Decoder(nn.Module):
    def __init__(self, state_dim, action_dim, output_dim, hidden_dim, num_layers):
        super(Decoder, self).__init__()
        self.num_layers = num_layers
        self.hidden_layers = nn.ModuleList()
        # Add first layer
        self.hidden_layers.append(nn.Linear(state_dim + action_dim, hidden_dim))
        # Add subsequent hidden layers
        for _ in range(num_layers - 1):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        # self.relu = nn.ReLU()
    
    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        for i in range(self.num_layers):
            x = F.leaky_relu(self.hidden_layers[i](x))
        x = self.output_layer(x)
        return x

# Define the Autoencoder
class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, state_dim, action_dim, output_dim, num_layers):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, state_dim, num_layers)
        self.decoder = Decoder(state_dim, action_dim, output_dim, hidden_dim, num_layers)

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
            loss = criterion(prediction, next_obs)
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
                loss = criterion(prediction, next_obs)
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
    plt.grid(False)
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
    output_dim = next_observations.shape[1]
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
    autoencoder = Autoencoder(input_dim, hidden_dim, state_dim, action_dim, output_dim, num_layers)
    train_autoencoder_with_validation(autoencoder, train_loader, val_loader, epochs, learning_rate, batch_size, state_dim, action_dim)
    evaluate_on_test(autoencoder, test_loader, batch_size, state_dim, action_dim)

if __name__ == "__main__":
    main()
