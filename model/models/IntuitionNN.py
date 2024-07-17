import torch
import torch.nn as nn
import torch.nn.functional as F

class IntuitionNN(nn.Module):
    def __init__(self, input_size, layer_sizes, intuition_size, num_labels):
        super(IntuitionNN, self).__init__()
        self.layers = nn.ModuleList()
        self.gates = nn.ModuleList()
        self.extra_input_layers = nn.ModuleList()
        self.initial_weights = []
        self.intuition_layer = nn.Linear(input_size, layer_sizes[-1])  # Match intuition size to last layer's output size
        self.intuition_coefficients = nn.Parameter(torch.zeros(layer_sizes[-1]))
        self.output_layer = nn.Linear(layer_sizes[-1], num_labels)

        for i in range(len(layer_sizes)):
            layer = nn.Linear(layer_sizes[i-1] if i > 0 else input_size, layer_sizes[i])
            self.layers.append(layer)
            self.initial_weights.append(layer.weight.clone().detach())
            if i > 1:
                extra_input_layer = nn.Linear(layer_sizes[i-2], layer_sizes[i])
                self.extra_input_layers.append(extra_input_layer)
                self.gates.append(nn.Linear(layer_sizes[i-1], layer_sizes[i]))  # Correct input size for the gate layer

    def forward(self, x, iteration):
        x_prev = None
        x_prev_prev = None
        intuition_output = torch.sigmoid(self.intuition_layer(x)) * self.intuition_coefficients
        outputs = []

        for i, layer in enumerate(self.layers):
            if i > 1 and x_prev_prev is not None and iteration % 2 == 1:
                extra_input = self.extra_input_layers[i-2](x_prev_prev)
                gate = torch.sigmoid(self.gates[i-2](x_prev))
                x = F.relu(layer(x) + gate * extra_input)
            else:
                x = F.relu(layer(x))
            
            outputs.append(x)
            x_prev_prev = x_prev
            x_prev = x

        x = self.output_layer(x)
        return x, intuition_output, outputs

    def compare_and_adjust(self, outputs, intuition_output):
        for i, layer in enumerate(self.layers):
            initial_weight = self.initial_weights[i]
            current_weight = layer.weight
            difference = current_weight - initial_weight
            adjustment = self.learn_from_difference(difference)
            layer.weight.data += adjustment

        intuition_adjustment = self.learn_from_intuition(outputs[-1], intuition_output)
        self.intuition_coefficients.data += intuition_adjustment

    def learn_from_difference(self, difference):
        return difference * 0.0001  # Adjustment factor

    def learn_from_intuition(self, output, intuition_output):
        # Ensure the tensors have compatible dimensions
        if output.size() != intuition_output.size():
            intuition_output = F.interpolate(intuition_output.unsqueeze(0), size=output.size(0), mode='nearest').squeeze(0)
        intuition_difference = (output - intuition_output).pow(2).mean()
        return intuition_difference * 0.0001  # Adjustment factor for intuition
