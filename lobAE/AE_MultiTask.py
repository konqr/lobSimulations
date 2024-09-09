import pandas as pd
import numpy as np
import random
import ast
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split

'''
What to do next:
1. Add quantile variable to position
'''
## Set random seed
def set_seed(seed=42):
    """Set all random seeds to a fixed value and take out any randomness from cuda kernels."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


## Data Preparation
def read_data(dataPath):
    data = pd.read_csv(dataPath).drop('Unnamed: 0',axis=1).rename(columns={'index': 'Seq'})
    data['Position'] = data['Position'].replace(np.nan, '{}')
    data['Position'] = data['Position'].apply(ast.literal_eval)
    return data


def prepare_data(data, max_positions=16):
    # Standardization
    data[['A2P', 'A1P', 'B1P', 'B2P', 'Price']] = data[['A2P', 'A1P', 'B1P', 'B2P', 'Price']] / 1e4
    data[['A2V', 'A1V', 'B1V', 'B2V', 'Volume']] = np.log(data[['A2V', 'A1V', 'B1V', 'B2V', 'Volume']])

    lob_cols = ['A2P','A1P','B1P','B2P','A2V','A1V','B1V','B2V']
    state_cols = ['Cash', 'Inventory']
    action_cols = ['Price', 'Volume', 'Type', 'Direction']

    # Flatten positions
    def flatten_position_vectorized(position_series, max_positions):
        def flatten_pos(pos):
            if len(pos) == 0:
                return np.zeros(2 * max_positions)
            else:
                pos_array = np.array(list(pos.values()), dtype=float)
                pos_array[:, 0] /= 1e4  # Scale prices
                pos_array[:, 1] = np.log(pos_array[:, 1])
                flat_position = pos_array.flatten()
                if len(flat_position) < 2 * max_positions:
                    flat_position = np.concatenate([flat_position, np.zeros(2 * max_positions - len(flat_position))])
                return flat_position[:2 * max_positions]
        
        tqdm.pandas(desc='apply')
        flattened = position_series.progress_apply(flatten_pos)
        return np.stack(flattened.values)

    flattened_positions = flatten_position_vectorized(data['Position'], max_positions)
    
    # Prepare observations and next observations
    observations = np.concatenate([data[state_cols].values, data[lob_cols].values, flattened_positions], axis=1)
    next_observations = data[lob_cols].shift(-1).iloc[:-1].values

    # Prepare actions
    actions = data[action_cols].iloc[:-1]
    actions = pd.get_dummies(actions, columns=['Type', 'Direction']).values

    observations = torch.tensor(observations[:-1], dtype=torch.float32)
    actions = torch.tensor(actions.astype(float), dtype=torch.float32)
    next_observations = torch.tensor(next_observations.astype(float), dtype=torch.float32)
    
    return observations, actions, next_observations


class CustomDataset(Dataset):
    def __init__(self, observations, actions, next_observations):
        self.observations = observations
        self.actions = actions
        self.next_observations = next_observations
    
    def __len__(self):
        return len(self.observations)
    
    def __getitem__(self, idx):
        return self.observations[idx], self.actions[idx], self.next_observations[idx]


# Standardisation within each batch
def batch_standardize(batch):
    observations, actions, next_observations = zip(*batch)
    
    observations = torch.stack(observations)
    actions = torch.stack(actions)
    next_observations = torch.stack(next_observations)
    
    # Standardize within the batch
    observations_mean = observations.mean(dim=0)
    observations_std = observations.std(dim=0)
    actions_mean = actions.mean(dim=0)
    actions_std = actions.std(dim=0)
    next_obs_mean = next_observations.mean(dim=0)
    next_obs_std = next_observations.std(dim=0)

    # Avoid division by zero
    observations_std[observations_std == 0] = 1
    actions_std[actions_std == 0] = 1
    next_obs_std[next_obs_std == 0] = 1
    
    standardized_observations = (observations - observations_mean) / observations_std
    standardized_actions = (actions - actions_mean) / actions_std
    standardized_next_observations = (next_observations - next_obs_mean) / next_obs_std

    standardized_actions = torch.nan_to_num(standardized_actions, nan=0.0)
    
    return standardized_observations, standardized_actions, standardized_next_observations, next_obs_mean, next_obs_std


# split the dataset into training, validating and testing
def split_dataset(observations, actions, next_observations, train_size=0.7, val_size=0.15):
    # Split into train + validation and test
    obs_train, obs_temp, act_train, act_temp, next_obs_train, next_obs_temp = train_test_split(
        observations, actions, next_observations, test_size=(1 - train_size), random_state=42, shuffle=False
    )
    
    # Split temp into validation and test
    val_size_adj = val_size / (1 - train_size)  # Adjust validation size based on remaining data
    obs_val, obs_test, act_val, act_test, next_obs_val, next_obs_test = train_test_split(
        obs_temp, act_temp, next_obs_temp, test_size=(1 - val_size_adj), random_state=42, shuffle=False
    )
    
    return obs_train, obs_val, obs_test, act_train, act_val, act_test, next_obs_train, next_obs_val, next_obs_test


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

    def forward(self, x):
        for layer in self.hidden_layers:
            x = F.leaky_relu(layer(x))
        return x

# Define the Decoder with configurable depth
class MultiTaskDecoder(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, num_layers, output_dims):
        super(MultiTaskDecoder, self).__init__()
        self.shared_layers = nn.ModuleList()
        self.shared_layers.append(nn.Linear(state_dim + action_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.shared_layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        # Separate output layers for price and volume
        self.price_decoder = nn.Linear(hidden_dim, output_dims['price'])
        self.volume_decoder = nn.Linear(hidden_dim, output_dims['volume'])
    
    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        for layer in self.shared_layers:
            x = F.leaky_relu(layer(x))
        
        # Separate predictions for price and volume
        price_pred = self.price_decoder(x)
        volume_pred = self.volume_decoder(x)
        
        # Combine predictions
        return torch.cat((price_pred, volume_pred), dim=1)

# Define the Autoencoder
class MultiTaskAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, state_dim, action_dim, output_dims, num_layers):
        super(MultiTaskAutoencoder, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, state_dim, num_layers)
        self.decoder = MultiTaskDecoder(state_dim, action_dim, hidden_dim, num_layers, output_dims)

    def forward(self, observation, action):
        state = self.encoder(observation)
        prediction = self.decoder(state, action)
        return prediction


## Model Training
def train_autoencoder_with_validation(autoencoder, train_loader, val_loader, epochs, learning_rate=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        autoencoder.train()
        total_train_loss = 0
        for obs, action, next_obs, _, _ in train_loader:
            optimizer.zero_grad()
            prediction = autoencoder(obs, action)
            # Separate loss for price and volume
            loss_price = criterion(prediction[:, :4], next_obs[:, :4])
            loss_volume = criterion(prediction[:, 4:], next_obs[:, 4:])
            loss = loss_price + loss_volume
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        autoencoder.eval()
        total_val_loss = 0
        with torch.no_grad():
            for obs, action, next_obs, _, _ in val_loader:
                prediction = autoencoder(obs, action)
                # Separate loss for price and volume
                loss_price = criterion(prediction[:, :4], next_obs[:, :4])
                loss_volume = criterion(prediction[:, 4:], next_obs[:, 4:])
                loss = loss_price + loss_volume
                total_val_loss += loss.item()
        
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

def evaluate_on_test(autoencoder, test_loader):
    autoencoder.eval()
    total_test_loss = 0
    criterion = nn.MSELoss()

    predictions = []
    actuals = []
    
    with torch.no_grad():
        for obs, action, next_obs, next_obs_mean, next_obs_std in test_loader:
            prediction = autoencoder(obs, action)
            loss = criterion(prediction, next_obs)
            total_test_loss += loss.item()
            restored_prediction = (prediction * next_obs_std) + next_obs_mean
            restored_actual = (next_obs * next_obs_std) + next_obs_mean
            restored_prediction[:, :4] = torch.round(restored_prediction[:, :4], decimals=2)
            restored_actual[:, :4] = torch.round(restored_actual[:, :4], decimals=2)
            restored_prediction[:, -4:] = torch.round(torch.exp(restored_prediction[:, -4:]))
            restored_actual[:, -4:] = torch.round(torch.exp(restored_actual[:, -4:]))
            predictions.append(restored_prediction.numpy())
            actuals.append(restored_actual.numpy())
    
    predictions = np.concatenate(predictions, axis=0)
    actuals = np.concatenate(actuals, axis=0)
    avg_test_loss = total_test_loss / len(test_loader)
    print('---')
    print(f'Test Loss: {avg_test_loss:.4f}')

    # Output the first few predictions and actuals for inspection
    for i in range(5):
        print('---')
        print(f"Prediction {i+1}: {predictions[i]}")
        print(f"Actual {i+1}: {actuals[i]}")


## Main Function
def main():
    set_seed(42)
    data = read_data('/Users/sw/Working Space/Python/nmdp_rl/data/dataset/AAPL_2019-01-02_dataset_2ls.csv')
    observations, actions, next_observations = prepare_data(data)
    obs_train, obs_val, obs_test, act_train, act_val, act_test, next_obs_train, next_obs_val, next_obs_test = split_dataset(
        observations, actions, next_observations
    )

    # Hyperparameters
    input_dim = observations.shape[1]
    hidden_dim = input_dim
    state_dim = 16
    action_dim = actions.shape[1]
    output_dims = {'price': 4, 'volume': 4}  # Output dimensions for price and volume
    num_layers = 8  # Adjust the depth of the network
    batch_size = 512
    epochs = 100

    # Prepare DataLoader
    train_dataset = CustomDataset(obs_train, act_train, next_obs_train)
    val_dataset = CustomDataset(obs_val, act_val, next_obs_val)
    test_dataset = CustomDataset(obs_test, act_test, next_obs_test)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=False, collate_fn=batch_standardize)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False, collate_fn=batch_standardize)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False, collate_fn=batch_standardize)

    # Instantiate and train the model
    autoencoder = MultiTaskAutoencoder(input_dim, hidden_dim, state_dim, action_dim, output_dims, num_layers)
    train_autoencoder_with_validation(autoencoder, train_loader, val_loader, epochs)
    evaluate_on_test(autoencoder, test_loader)

if __name__ == "__main__":
    main()
