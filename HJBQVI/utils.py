import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import json
from datetime import datetime
import os
from collections import OrderedDict

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

            network = parts[0]
            metric_type = 'loss' if 'loss' in key else 'acc'

            # Add network to tracking if it's new
            if network not in self.networks:
                self.networks.append(network)


            # Initialize list for this metric if it doesn't exist
            if key not in self.losses:
                self.losses[key] = []
                self.tracked_metrics[key] = {'network': network, 'type': metric_type}

            # Append the value
            self.losses[key].append(value)

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
                return [None]*len(models.keys())

            # Get the latest timestamp
            meta_files.sort(reverse=True)
            latest_meta = meta_files[0]
            meta_path = os.path.join(self.model_dir, latest_meta)
        else:
            # Use specified timestamp
            suffix = f"_epoch_{epoch}" if epoch >= 0 else "_final"
            meta_path = os.path.join(
                self.model_dir,
                f"model_metadata{suffix}_{timestamp}.json"
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
            return [None]*len(models.keys())
        except Exception as e:
            print(f"Error loading models: {e}")
            return [None]*len(models.keys())

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