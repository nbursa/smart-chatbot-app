import torch
import torch.nn as nn
import torch.nn.functional as F

class IntuitionNN(nn.Module):
    def __init__(self, input_size, layer_sizes, intuition_size):
        super(IntuitionNN, self).__init__()
        self.layers = nn.ModuleList()
        self.initial_weights = []
        self.intuition_layer = nn.Linear(input_size, intuition_size)
        self.intuition_coefficients = torch.zeros(intuition_size)

        for i in range(len(layer_sizes)):
            layer = nn.Linear(layer_sizes[i-1] if i > 0 else input_size, layer_sizes[i])
            self.layers.append(layer)
            self.initial_weights.append(layer.weight.clone().detach())
            if i > 1:
                extra_input_layer = nn.Linear(layer_sizes[i-2], layer_sizes[i])
                self.add_module(f'extra_input_layer_{i}', extra_input_layer)

    def forward(self, x, iteration):
        x_prev_prev = None
        intuition_output = self.intuition_layer(x) * self.intuition_coefficients
        for i, layer in enumerate(self.layers):
            if i > 1 and x_prev_prev is not None:
                extra_input_output = F.relu(getattr(self, f'extra_input_layer_{i}')(x_prev_prev))
                pre_computed = extra_input_output / 2
                x = F.relu(layer(x)) + pre_computed
            else:
                x = F.relu(layer(x))
            x_prev_prev = x_prev if i > 0 else x
            x_prev = x
        return x, intuition_output

    def train_step(self, embeddings, labels, optimizer, criterion):
        optimizer.zero_grad()
        outputs, _ = self(embeddings, 0)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
