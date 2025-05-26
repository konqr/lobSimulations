###This segment of the code is taken from https://github.com/alirezakazemipour/Discrete-SAC-PyTorch/blob/main/Brain/model.py for proof-of-concept purposes


from abc import ABC
from torch import nn
from torch.nn import functional as F
from torch.distributions.categorical import Categorical


def conv_shape(input, kernel_size, stride, padding=0):
    return (input + 2 * padding - kernel_size) // stride + 1


class QValueNetwork(nn.Module, ABC):
    def __init__(self, state_shape, n_actions, activation=nn.Identity):
        super().__init__()
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.layer_1=nn.Linear(in_features=self.state_shape, out_features=64)
        self.layer_2 = nn.Linear(in_features=64, out_features=64)
        self.q_value = nn.Linear(in_features=64, out_features=self.n_actions)
        self.activation=activation

        #Initialize
        nn.init.kaiming_normal_(self.layer_1.weight, nonlinearity="relu")
        nn.init.zeros_(self.layer_1.bias)
        nn.init.kaiming_normal_(self.layer_2.weight, nonlinearity="relu")
        nn.init.zeros_(self.layer_2.bias)
        nn.init.xavier_uniform_(self.q_value.weight)
        nn.init.zeros_(self.q_value.bias)

    def forward(self, states):
        x = F.relu(self.layer_1(states))
        y = F.relu(self.layer_2(x))
        output = self.activation(self.q_value(y))
        return output


class PolicyNetwork(nn.Module, ABC):
    def __init__(self, state_shape, n_actions, activation=nn.Softmax(dim=1)):
        super().__init__()
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.layer_1=nn.Linear(in_features=self.state_shape, out_features=64)
        self.layer_2 = nn.Linear(in_features=64, out_features=64)
        self.logits = nn.Linear(in_features=64, out_features=self.n_actions)
        self.activation=activation

        nn.init.kaiming_normal_(self.layer_1.weight, nonlinearity="relu")
        nn.init.zeros_(self.layer_1.bias)
        nn.init.kaiming_normal_(self.layer_2.weight, nonlinearity="relu")
        nn.init.zeros_(self.layer_2.bias)
        nn.init.xavier_uniform_(self.logits.weight)
        nn.init.zeros_(self.logits.bias)

    def forward(self, states):
        x = F.relu(self.layer_1(states))
        y = F.relu(self.layer_2(x))
        logits = self.logits(y)
        probs = self.activation(logits)
        return probs