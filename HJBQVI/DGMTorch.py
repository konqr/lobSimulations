import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import json
from datetime import datetime
import os
from collections import OrderedDict

class LSTMLayer(nn.Module):
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
        Z = self.trans1(torch.add(torch.add(torch.matmul(X, self.Uz), torch.matmul(S, self.Wz)), self.bz))
        G = self.trans1(torch.add(torch.add(torch.matmul(X, self.Ug), torch.matmul(S, self.Wg)), self.bg))
        R = self.trans1(torch.add(torch.add(torch.matmul(X, self.Ur), torch.matmul(S, self.Wr)), self.br))

        H = self.trans2(torch.add(torch.add(torch.matmul(X, self.Uh), torch.matmul(torch.mul(S, R), self.Wh)), self.bh))

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
                "relu": torch.relu
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


class DGMNet(nn.Module):
    def __init__(self, layer_width, n_layers, input_dim, output_dim = 1, final_trans=None, typeNN = 'LSTM'):
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
        self.initial_layer = DenseLayer(layer_width, input_dim+1, transformation="tanh")

        # Intermediate LSTM layers
        self.n_layers = n_layers
        if typeNN == 'LSTM':
            self.LayerList = nn.ModuleList([
                LSTMLayer(layer_width, input_dim+1) for _ in range(self.n_layers)
            ])
        elif typeNN == 'Dense':
            self.LayerList = nn.ModuleList([
                DenseLayer(layer_width, layer_width, transformation="tanh") for _ in range(self.n_layers)
            ])
        else:
            raise Exception('typeNN ' + typeNN + ' not supported. Choose one of Dense or LSTM')
        self.typeNN = typeNN
        # Final layer as fully connected with a single output (function value)
        self.final_layer = DenseLayer(output_dim, layer_width, transformation=final_trans)

    def forward(self, t, x):
        '''
        Args:
            t: sampled time inputs
            x: sampled space inputs

        Run the DGM model and obtain fitted function value at the inputs (t,x)
        '''
        # Define input vector as time-space pairs

        X = torch.cat([t, x], 1)
        X = MinMaxScaler().fit_transform(X)
        # Call initial layer
        S = self.initial_layer(X)

        # Call intermediate LSTM layers
        for layer in self.LayerList:
            if self.typeNN == 'LSTM':
                S = layer(S, X)
            elif self.typeNN == 'Dense':
                S = layer(S)

        # Call final layer
        result = self.final_layer(S)

        return result

class PIANet(nn.Module):
    def __init__(self, layer_width, n_layers, input_dim, num_classes, final_trans=None, typeNN = 'LSTM'):
        '''
        Args:
            layer_width:
            n_layers:    number of intermediate LSTM layers
            input_dim:   spatial dimension of input data (EXCLUDES time dimension)
            num_classes: number of output classes
            final_trans: transformation used in final layer

        Returns: customized PyTorch model object representing DGM neural network
        '''
        super(PIANet, self).__init__()

        # Initial layer as fully connected
        # NOTE: to account for time inputs we use input_dim+1 as the input dimensionality
        self.initial_layer = DenseLayer(layer_width, input_dim+1, transformation="tanh")

        # Intermediate LSTM layers
        self.n_layers = n_layers
        if typeNN == 'LSTM':
            self.LayerList = nn.ModuleList([
                LSTMLayer(layer_width, input_dim+1) for _ in range(self.n_layers)
            ])
        elif typeNN == 'Dense':
            self.LayerList = nn.ModuleList([
                DenseLayer(layer_width, layer_width, transformation="tanh") for _ in range(self.n_layers)
            ])
        else:
            raise Exception('typeNN ' + typeNN + ' not supported. Choose one of Dense or LSTM')
        self.typeNN = typeNN
        # Final layer as fully connected with multiple outputs (function values)
        self.final_layer = DenseLayer(num_classes, layer_width, transformation=final_trans)

    def forward(self, t, x):
        '''
        Args:
            t: sampled time inputs
            x: sampled space inputs

        Run the DGM model and obtain fitted function value at the inputs (t,x)
        '''
        # Define input vector as time-space pairs
        X = torch.cat([t, x], 1)
        X = MinMaxScaler().fit_transform(X)
        # Call initial layer
        S = self.initial_layer(X)

        # Call intermediate LSTM layers
        for layer in self.LayerList:
            if self.typeNN == 'LSTM':
                S = layer(S, X)
            elif self.typeNN == 'Dense':
                S = layer(S)
        # Call final layer
        result = self.final_layer(S)

        # Get argmax and result
        op = torch.argmax(result, 1)
        op = op.reshape(op.shape[0], 1).float()

        return op, result

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

class MinMaxScaler:
    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range
        self.data_min_ = None
        self.data_max_ = None

    def fit(self, X):
        """Compute min/max values for scaling along each feature (column)"""
        self.data_min_ = X.min(dim=0).values
        self.data_max_ = X.max(dim=0).values
        return self

    def transform(self, X):
        """Apply min-max scaling using precomputed statistics"""
        denominator = self.data_max_ - self.data_min_
        denominator = torch.where(denominator == 0, torch.ones_like(denominator), denominator)

        scale = (self.feature_range[1] - self.feature_range[0]) / denominator
        min_ = self.feature_range[0] - self.data_min_ * scale

        return X * scale + min_

    def fit_transform(self, X):
        """Convenience method for fit+transform"""
        return self.fit(X).transform(X)

    def inverse_transform(self, X_scaled):
        """Reverse the scaling operation"""
        denominator = self.data_max_ - self.data_min_
        denominator = torch.where(denominator == 0, torch.ones_like(denominator), denominator)

        scale = (self.feature_range[1] - self.feature_range[0]) / denominator
        return (X_scaled - (self.feature_range[0] - self.data_min_ * scale)) / scale

class TrainingLogger:
    def __init__(self, log_dir='./logs', layer_widths=[20]*3, n_layers=[5]*3, label=None):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.hyperparams = {
            'layer_widths': layer_widths,
            'n_layers': n_layers
        }

        # Initialize loss dictionaries
        self.losses = {
            'hyperparams': self.hyperparams,
            'epochs': []
        }

        # Initialize tracking variables
        self.networks = []
        self.tracked_metrics = {}

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.label = '' if label is None else str(label)
        self.current_epoch = 0

    def log_losses(self, **metrics):
        """
        Log losses and accuracies flexibly

        Args:
            **metrics: Dictionary of metrics to log (e.g., phi_loss=0.1, u_loss=0.2, acc_u=0.85)
        """
        self.current_epoch += 1
        self.losses['epochs'].append(self.current_epoch)

        # Process each provided metric
        for key, value in metrics.items():
            # Extract network name and metric type
            parts = key.split('_')

            if len(parts) == 1:
                # Just the network name (like 'phi') implies it's a loss
                network = parts[0]
                metric_type = 'loss'
            else:
                # Format like 'acc_u' or 'u_loss'
                if parts[0] == 'acc':
                    network = parts[1]
                    metric_type = 'acc'
                else:
                    network = parts[0]
                    metric_type = parts[1]

            # Add network to tracking if it's new
            if network not in self.networks:
                self.networks.append(network)

            # Create metric key
            metric_key = f"{network}_{metric_type}"

            # Initialize list for this metric if it doesn't exist
            if metric_key not in self.losses:
                self.losses[metric_key] = []
                self.tracked_metrics[metric_key] = {'network': network, 'type': metric_type}

            # Append the value
            self.losses[metric_key].append(value)

    def save_logs(self):
        """Save logs to file"""
        log_file = os.path.join(self.log_dir, f"training_logs_{self.timestamp}_{self.label}.json")
        with open(log_file, 'w') as f:
            json.dump(self.losses, f)
        return log_file

    def plot_losses(self, show=True, save=True):
        """Plot the loss curves with linear and log scales"""
        # Count how many metrics we have to plot
        loss_metrics = [m for m in self.tracked_metrics.keys() if 'loss' in m]
        acc_metrics = [m for m in self.tracked_metrics.keys() if 'acc' in m]
        total_metrics = len(loss_metrics) + len(acc_metrics)

        # Calculate rows needed (each metric gets 2 plots side by side)
        rows = max(total_metrics, 1)

        plt.figure(figsize=(18, 4 * rows))  # Adjust height based on rows

        epochs = self.losses['epochs']
        plot_idx = 1

        # Define colors for each network
        colors = {'phi': 'b', 'u': 'g', 'd': 'r'}

        # Plot losses first
        for metric in loss_metrics:
            network = self.tracked_metrics[metric]['network']
            color = colors.get(network, 'k')  # Default to black if network not in colors dict

            # Linear scale
            plt.subplot(rows, 2, plot_idx)
            plt.plot(epochs, self.losses[metric], f'{color}-')
            plt.title(f'{network.capitalize()} Loss (Linear)')
            plt.ylabel('Loss')
            plt.grid(True)

            # Log scale
            plt.subplot(rows, 2, plot_idx + 1)
            plt.semilogy(epochs, self.losses[metric], f'{color}-')
            plt.title(f'{network.capitalize()} Loss (Log Scale)')
            plt.ylabel('Log Loss')
            plt.grid(True)

            plot_idx += 2

        # Then plot accuracies
        for metric in acc_metrics:
            network = self.tracked_metrics[metric]['network']
            color = colors.get(network, 'k')

            # Linear scale
            plt.subplot(rows, 2, plot_idx)
            plt.plot(epochs, self.losses[metric], f'{color}.-')
            plt.title(f'{network.capitalize()} Accuracy (Linear)')
            plt.ylabel('Accuracy')
            plt.grid(True)

            # Log scale
            plt.subplot(rows, 2, plot_idx + 1)
            plt.semilogy(epochs, self.losses[metric], f'{color}.-')
            plt.title(f'{network.capitalize()} Accuracy (Log Scale)')
            plt.ylabel('Log Accuracy')
            plt.grid(True)

            plot_idx += 2

        # Add epoch labels to bottom plots
        for i in range(max(1, (plot_idx - 2)), plot_idx):
            plt.subplot(rows, 2, i)
            plt.xlabel('Epochs')

        plt.tight_layout(pad=3.0)  # Add padding between subplots

        if save:
            plot_file = os.path.join(self.log_dir, f"loss_plot_{self.timestamp}_{self.label}.png")
            plt.savefig(plot_file)
            print(f"Loss plot saved to {plot_file}")

        if show:
            plt.show()

        return plt.gcf()


# Model management helper class
class ModelManager:
    def __init__(self, model_dir='./models', label=None):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.label = '' if label is None else str(label)

    def save_models(self, epoch=-1, **models):
        """Save models to files

        Args:
            epoch (int): Current epoch number or -1 for final model
            **models: Dictionary of models to save, e.g. phi=model_phi, u=model_u, d=model_d
        """
        # Create epoch-specific or final save
        suffix = f"_epoch_{epoch}" if epoch >= 0 else "_final"

        # Save model states and collect paths
        metadata = {
            'timestamp': self.timestamp,
            'epoch': epoch,
            'models': {}
        }

        for model_name, model in models.items():
            if model is None:
                continue

            model_path = os.path.join(
                self.model_dir,
                f"model_{model_name}{suffix}_{self.timestamp}_{self.label}.pt"
            )
            torch.save(model.state_dict(), model_path)
            metadata['models'][model_name] = model_path

        # Save model metadata
        meta_path = os.path.join(
            self.model_dir,
            f"model_metadata{suffix}_{self.timestamp}_{self.label}.json"
        )
        with open(meta_path, 'w') as f:
            json.dump(metadata, f)

        return metadata

    def load_models(self, timestamp=None, epoch=-1, **models):
        """Load models from files

        Args:
            timestamp (str): Specific timestamp to load or None for latest
            epoch (int): Specific epoch to load or -1 for final model
            **models: Dictionary of model objects to load into, e.g. phi=model_phi, u=model_u

        Returns:
            dict: Dictionary of loaded models with same keys as input
        """
        # Find latest model if timestamp not provided
        if timestamp is None:
            # List all metadata files and find the latest one
            meta_files = [f for f in os.listdir(self.model_dir) if f.startswith("model_metadata")]
            if not meta_files:
                print("No saved models found.")
                return None

            # Get the latest timestamp
            meta_files.sort(reverse=True)
            latest_meta = meta_files[0]
            meta_path = os.path.join(self.model_dir, latest_meta)
        else:
            # Use specified timestamp
            suffix = f"_epoch_{epoch}" if epoch >= 0 else "_final"
            meta_path = os.path.join(
                self.model_dir,
                f"model_metadata{suffix}_{timestamp}_{self.label}.json"
            )

        # Load metadata
        try:
            with open(meta_path, 'r') as f:
                metadata = json.load(f)

            loaded_models = {}
            device = 'cpu' if not torch.cuda.is_available() else 'cuda'

            # Load each model if provided and available in metadata
            for model_name, model in models.items():
                if model is None or model_name not in metadata['models']:
                    loaded_models[model_name] = None
                    continue

                try:
                    state_dict = torch.load(
                        metadata['models'][model_name],
                        map_location=torch.device(device)
                    )

                    # Handle potential module prefix differences
                    if not any(k.startswith('module.') for k in state_dict.keys()) and \
                            any(k.startswith('module.') for k in model.state_dict().keys()):
                        # Add module prefix
                        new_state_dict = OrderedDict()
                        for k, v in state_dict.items():
                            new_state_dict["module." + k] = v
                        state_dict = new_state_dict
                    elif any(k.startswith('module.') for k in state_dict.keys()) and \
                            not any(k.startswith('module.') for k in model.state_dict().keys()):
                        # Remove module prefix
                        new_state_dict = OrderedDict()
                        for k, v in state_dict.items():
                            new_state_dict[k.replace('module.', '')] = v
                        state_dict = new_state_dict

                    model.load_state_dict(state_dict)
                    loaded_models[model_name] = model
                except Exception as e:
                    print(f"Error loading {model_name}: {e}")
                    loaded_models[model_name] = None

            print(f"Models loaded successfully from {meta_path}")
            return loaded_models

        except FileNotFoundError:
            print(f"Model files not found at {meta_path}")
            return None
        except Exception as e:
            print(f"Error loading models: {e}")
            return None

def get_gpu_specs():
    """Print detailed information about available CUDA devices in PyTorch."""

    print("=" * 40)
    print("PYTORCH GPU INFORMATION")
    print("=" * 40)

    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available. No GPU detected by PyTorch.")
        return

    # Get the number of GPUs
    gpu_count = torch.cuda.device_count()
    print(f"Number of available GPUs: {gpu_count}")

    # Current device
    current_device = torch.cuda.current_device()
    print(f"Current active device: {current_device}")

    # Get PyTorch CUDA version
    cuda_version = torch.version.cuda
    print(f"PyTorch CUDA version: {cuda_version}")

    # Loop through each device and get its properties
    for i in range(gpu_count):
        print(f"\nGPU {i} specifications:")
        print("-" * 30)

        # Get device properties
        prop = torch.cuda.get_device_properties(i)

        # Print device name and specifications
        print(f"Name: {prop.name}")
        print(f"Compute capability: {prop.major}.{prop.minor}")
        print(f"Total memory: {prop.total_memory / 1024**3:.2f} GB")

        # Additional properties
        print(f"Multi-processor count: {prop.multi_processor_count}")

    # Check memory usage of current device
    print("\nCurrent GPU Memory Usage:")
    print("-" * 30)
    print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"Cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

import json
import matplotlib.pyplot as plt
import os
from datetime import datetime

def plot_losses_from_json(json_path, show=True, save=True):
    """Load loss data from JSON and create dual-scale plots"""
    # Load data
    with open(json_path, 'r') as f:
        losses = json.load(f)

    # Validate required keys
    required_keys = ['phi', 'u', 'd']
    for key in required_keys:
        if key not in losses:
            raise ValueError(f"Missing required key in JSON: {key}")

    # Create figure
    plt.figure(figsize=(18, 12))
    epochs = range(1, len(losses['phi']) + 1)

    # Plotting function
    def plot_pair(position, data, color, label, is_accuracy=False):
        # Linear scale
        plt.subplot(5, 2, position*2-1)
        plt.plot(epochs, data, f'{color}-' if not is_accuracy else f'{color}.')
        plt.title(f'{label} (Linear)')
        plt.ylabel('Accuracy' if is_accuracy else 'Loss')
        plt.grid(True)

        # Log scale
        plt.subplot(5, 2, position*2)
        plot_data = data if not is_accuracy else [x + 1e-8 for x in data]  # Avoid log(0)
        plt.semilogy(epochs, plot_data, f'{color}-' if not is_accuracy else f'{color}.')
        plt.title(f'{label} (Log Scale)')
        plt.ylabel('Log Accuracy' if is_accuracy else 'Log Loss')
        plt.grid(True)

    # Plot all components
    plot_pair(1, losses['phi'], 'b', 'Value Function Loss')
    plot_pair(2, losses['u'], 'g', 'Control Function Loss')
    #plot_pair(3, losses['acc_u'], 'g', 'Control Function Accuracy', is_accuracy=True)
    plot_pair(4, losses['d'], 'r', 'Decision Function Loss')
    #plot_pair(5, losses['acc_d'], 'r', 'Decision Function Accuracy', is_accuracy=True)

    # Final formatting
    plt.tight_layout(pad=3.0)
    plt.subplot(5, 2, 9)
    plt.xlabel('Epochs')
    plt.subplot(5, 2, 10)
    plt.xlabel('Epochs')

    # Save/output
    if save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = os.path.join(os.path.dirname(json_path), f"loss_plot_{timestamp}.png")
        plt.savefig(plot_path)
        print(f"Plot saved to: {plot_path}")

    if show:
        plt.show()

    return plt.gcf()

# Example usage
# plot_losses_from_json('D:\\PhD\\results - hjbqvi\\training_logs_20250317_120356.json')
