import pandas as pd
import numpy as np
import random
import ast
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm
from sklearn.model_selection import train_test_split


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
    
    # lob_cols = ['A2P', 'A2V', 'A1P', 'A1V', 'B1P', 'B1V', 'B2P', 'B2V']
    lob_cols = ['A2P', 'A1P', 'B1P', 'B2P', 'A2V', 'A1V', 'B1V', 'B2V']
    state_cols = ['Cash', 'Inventory']
    action_cols = ['Price', 'Volume', 'Type', 'Direction']

    # Flatten positions
    def flatten_position_vectorized(position_series, max_positions):
        def flatten_pos(pos):
            if len(pos) == 0:
                return np.zeros(3 * max_positions)
            else:
                pos_array = np.array(list(pos.values()), dtype=float)
                pos_array[:, 0] /= 1e4  # Scale prices
                pos_array[:, 1] = np.log(pos_array[:, 1])
                pos_array[:, 2] = pos_array[:, 3]
                pos_array[:, 2] = np.log(pos_array[:, 2] + 1)
                pos_array = pos_array[:, :-1]
                flat_position = pos_array.flatten()
                if len(flat_position) < 3 * max_positions:
                    flat_position = np.concatenate([flat_position, np.zeros(3 * max_positions - len(flat_position))])
                return flat_position[:3 * max_positions]
        
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

def evaluate_on_test(autoencoder, test_loader, batch_size, state_dim, action_dim):
    autoencoder.eval()
    total_test_loss = 0
    criterion = nn.MSELoss()

    predictions = []
    actuals = []
    
    with torch.no_grad():
        prev_state = torch.zeros((batch_size, state_dim))
        prev_action = torch.zeros((batch_size, action_dim))
        for obs, action, next_obs, next_obs_mean, next_obs_std in test_loader:
            prediction, state = autoencoder(obs, action, prev_state, prev_action)
            loss = criterion(prediction, next_obs)
            total_test_loss += loss.item()
            prev_state = state.detach()
            prev_action = action.detach()
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
