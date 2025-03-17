import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import json
from datetime import datetime
import os

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
    def __init__(self, layer_width, n_layers, input_dim, final_trans=None, typeNN = 'LSTM'):
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
        self.final_layer = DenseLayer(1, layer_width, transformation=final_trans)

    def forward(self, t, x):
        '''
        Args:
            t: sampled time inputs
            x: sampled space inputs

        Run the DGM model and obtain fitted function value at the inputs (t,x)
        '''
        # Define input vector as time-space pairs
        X = torch.cat([t, x], 1)

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

# Helper class for loss tracking and visualization
class TrainingLogger:
    def __init__(self, log_dir='./logs', layer_widths = [20]*3, n_layers= [5]*3, label=None):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.hyperparams = {'layer_widths': layer_widths,
                            'n_layers': n_layers}
        # Initialize loss dictionaries
        self.losses = {
            'hyperparams' : self.hyperparams,
            'phi': [],
            'd': [],
            'u': [],
            'acc_u' : [],
            'acc_d' : []
        }
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.label = ''
        if label: self.label = str(label)

    def log_losses(self, phi_loss, d_loss, u_loss, acc_d, acc_u):
        """Log losses for each epoch"""
        self.losses['phi'].append(phi_loss)
        self.losses['d'].append(d_loss)
        self.losses['u'].append(u_loss)
        self.losses['acc_d'].append(acc_d)
        self.losses['acc_u'].append(acc_u)

    def save_logs(self):
        """Save logs to file"""
        log_file = os.path.join(self.log_dir, f"training_logs_{self.timestamp}_{self.label}.json")
        with open(log_file, 'w') as f:
            json.dump(self.losses, f)
        return log_file

    def plot_losses(self, show=True, save=True):
        """Plot the loss curves with linear and log scales"""
        plt.figure(figsize=(18, 12))  # Wider figure for two columns

        epochs = range(1, len(self.losses['phi']) + 1)

        # Value Function
        plt.subplot(5, 2, 1)
        plt.plot(epochs, self.losses['phi'], 'b-')
        plt.title('Value Function Loss (Linear)')
        plt.ylabel('Loss')
        plt.grid(True)

        plt.subplot(5, 2, 2)
        plt.semilogy(epochs, self.losses['phi'], 'b-')
        plt.title('Value Function Loss (Log Scale)')
        plt.ylabel('Log Loss')
        plt.grid(True)

        # Control Function Loss
        plt.subplot(5, 2, 3)
        plt.plot(epochs, self.losses['u'], 'g-')
        plt.title('Control Function Loss (Linear)')
        plt.ylabel('Loss')
        plt.grid(True)

        plt.subplot(5, 2, 4)
        plt.semilogy(epochs, self.losses['u'], 'g-')
        plt.title('Control Function Loss (Log Scale)')
        plt.ylabel('Log Loss')
        plt.grid(True)

        # Control Function Accuracy
        plt.subplot(5, 2, 5)
        plt.plot(epochs, self.losses['acc_u'], 'g.')
        plt.title('Control Function Acc (Linear)')
        plt.ylabel('Accuracy')
        plt.grid(True)

        plt.subplot(5, 2, 6)
        plt.semilogy(epochs, self.losses['acc_u'], 'g.')
        plt.title('Control Function Acc (Log Scale)')
        plt.ylabel('Log Accuracy')
        plt.grid(True)

        # Decision Function Loss
        plt.subplot(5, 2, 7)
        plt.plot(epochs, self.losses['d'], 'r-')
        plt.title('Decision Function Loss (Linear)')
        plt.ylabel('Loss')
        plt.grid(True)

        plt.subplot(5, 2, 8)
        plt.semilogy(epochs, self.losses['d'], 'r-')
        plt.title('Decision Function Loss (Log Scale)')
        plt.ylabel('Log Loss')
        plt.grid(True)

        # Decision Function Accuracy
        plt.subplot(5, 2, 9)
        plt.plot(epochs, self.losses['acc_d'], 'r-')
        plt.title('Decision Function Acc (Linear)')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.grid(True)

        plt.subplot(5, 2, 10)
        plt.semilogy(epochs, self.losses['acc_d'], 'r-')
        plt.title('Decision Function Acc (Log Scale)')
        plt.xlabel('Epochs')
        plt.ylabel('Log Accuracy')
        plt.grid(True)

        plt.tight_layout(pad=3.0)  # Add more padding between subplots

        if save:
            plot_file = os.path.join(self.log_dir, f"loss_plot_{self.timestamp}_{self.label}.png")
            plt.savefig(plot_file)
            print(f"Loss plot saved to {plot_file}")

        if show:
            plt.show()

        return plt.gcf()


# Model management helper class
class ModelManager:
    def __init__(self, model_dir='./models', label = None):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.label = ''
        if label: self.label = str(label)

    def save_models(self, model_phi, model_d, model_u, epoch=-1):
        """Save models to files"""
        # Create epoch-specific or final save
        suffix = f"_epoch_{epoch}" if epoch >= 0 else "_final"

        # Save paths
        phi_path = os.path.join(self.model_dir, f"model_phi{suffix}_{self.timestamp}_{self.label}.pt")
        d_path = os.path.join(self.model_dir, f"model_d{suffix}_{self.timestamp}_{self.label}.pt")
        u_path = os.path.join(self.model_dir, f"model_u{suffix}_{self.timestamp}_{self.label}.pt")

        # Save model states
        torch.save(model_phi.state_dict(), phi_path)
        torch.save(model_d.state_dict(), d_path)
        torch.save(model_u.state_dict(), u_path)

        # Save model metadata
        metadata = {
            'timestamp': self.timestamp,
            'epoch': epoch,
            'models': {
                'phi': phi_path,
                'd': d_path,
                'u': u_path
            }
        }

        meta_path = os.path.join(self.model_dir, f"model_metadata{suffix}_{self.timestamp}_{self.label}.json")
        with open(meta_path, 'w') as f:
            json.dump(metadata, f)

        return metadata

    def load_models(self, model_phi, model_d, model_u, timestamp=None, epoch=-1):
        """Load models from files"""
        # Find latest model if timestamp not provided
        if timestamp is None:
            # List all metadata files and find the latest one
            meta_files = [f for f in os.listdir(self.model_dir) if f.startswith("model_metadata")]
            if not meta_files:
                print("No saved models found.")
                return None, None, None

            # Get the latest timestamp
            meta_files.sort(reverse=True)
            latest_meta = meta_files[0]
            meta_path = os.path.join(self.model_dir, latest_meta)
        else:
            # Use specified timestamp
            suffix = f"_epoch_{epoch}" if epoch >= 0 else "_final"
            meta_path = os.path.join(self.model_dir, f"model_metadata{suffix}_{timestamp}.json")

        # Load metadata
        try:
            with open(meta_path, 'r') as f:
                metadata = json.load(f)

            # Load model states
            model_phi.load_state_dict(torch.load(metadata['models']['phi']))
            model_d.load_state_dict(torch.load(metadata['models']['d']))
            model_u.load_state_dict(torch.load(metadata['models']['u']))

            print(f"Models loaded successfully from {meta_path}")
            return model_phi, model_d, model_u
        except FileNotFoundError:
            print(f"Model files not found at {meta_path}")
            return None, None, None
        except Exception as e:
            print(f"Error loading models: {e}")
            return None, None, None

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
