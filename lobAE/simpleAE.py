import pandas as pd
import numpy as np
import ast
import torch
import torch.nn as nn
import torch.optim as optim

'''
What to do next:
1. Add quantile variable to position
2. Standardization
'''

## Data Preparation
def read_data(dataPath):
    data = pd.read_csv(dataPath).drop('Unnamed: 0',axis=1).rename(columns={'index': 'Seq'})
    data['Position'] = data['Position'].replace(np.nan, '{}')
    data['Position'] = data['Position'].apply(ast.literal_eval)
    return data

def prepare_data(data, max_positions=16):
    lob_cols = ['A2P', 'A2V', 'A1P', 'A1V', 'B1P', 'B1V', 'B2P', 'B2V']
    state_cols = ['Cash', 'Inventory']
    action_cols = ['Price', 'Volume', 'Type', 'Direction']

    def flatten_position(position_dict, max_positions):
        flat_position = []
        for order_id, (price, volume) in position_dict.items():
            flat_position.append([price, volume])
        # Pad with zeros if fewer than max_positions, or truncate if more
        flat_position = flat_position[:max_positions] + [[0, 0]] * (max_positions - len(flat_position))
        return np.array(flat_position).flatten()

    observations = []
    actions = []
    next_observations = []

    for i in range(len(data) - 1):
        current_lob = data.loc[i, lob_cols].values
        current_position = flatten_position(data.loc[i, 'Position'], max_positions)
        current_state = data.loc[i, state_cols].values

        next_lob = data.loc[i + 1, lob_cols].values

        action = data.loc[i, action_cols].values

        current_obs = np.concatenate([current_lob, current_position, current_state])
        next_obs = np.concatenate([next_lob])

        observations.append(current_obs)
        actions.append(action)
        next_observations.append(next_obs)

    return np.array(observations), np.array(actions), np.array(next_observations)


## Model Definiton
# Define the Encoder
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return x

# Define the Decoder
class Decoder(nn.Module):
    def __init__(self, state_dim, action_dim, output_dim, hidden_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
    
    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Define the Autoencoder
class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, state_dim, action_dim, output_dim):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim)
        self.decoder = Decoder(state_dim, action_dim, output_dim, hidden_dim)
    
    def forward(self, observation, action):
        state = self.encoder(observation)
        prediction = self.decoder(state, action)
        return prediction

## Model Training
def train_autoencoder(autoencoder, observations, actions, next_observations, epochs=100, batch_size=32, learning_rate=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)
    
    dataset = list(zip(observations, actions, next_observations))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    for epoch in range(epochs):
        total_loss = 0
        for obs, action, next_obs in dataloader:
            obs = torch.tensor(obs, dtype=torch.float32)
            action = torch.tensor(action, dtype=torch.float32)
            next_obs = torch.tensor(next_obs, dtype=torch.float32)
            
            optimizer.zero_grad()
            prediction = autoencoder(obs, action)
            loss = criterion(prediction, next_obs)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f'Epoch {epoch+1}, Loss: {total_loss/len(dataloader)}')

def main():
    data = read_data('/Users/sw/Working Space/Python/nmdp_rl/data/dataset/AAPL_2019-01-02_dataset_2ls.csv')
    observations, actions, next_observations = prepare_data(data)
    # Hyperparameters
    input_dim = len(observations[0])  # Adjust this based on actual input dimensions
    hidden_dim = input_dim
    state_dim = 16
    action_dim = len(actions[0])  # Adjust this based on actual action dimensions
    output_dim = len(next_observations[0])  # Adjust this based on actual output dimensions
    # Instantiate and train the model
    autoencoder = Autoencoder(input_dim, hidden_dim, state_dim, action_dim, output_dim)
    train_autoencoder(autoencoder, observations, actions, next_observations)

if __name__ == "__main__":
    main()
