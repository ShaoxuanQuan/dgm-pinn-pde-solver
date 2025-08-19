import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DGMNet(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_layers, lb, ub):
        super(DGMNet, self).__init__()

        self.register_buffer('lb', lb)
        self.register_buffer('ub', ub)

        self.W1 = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'Uz': nn.Linear(input_dim, hidden_dim), 'Wz': nn.Linear(hidden_dim, hidden_dim),
                'Ug': nn.Linear(input_dim, hidden_dim), 'Wg': nn.Linear(hidden_dim, hidden_dim),
                'Ur': nn.Linear(input_dim, hidden_dim), 'Wr': nn.Linear(hidden_dim, hidden_dim),
                'Uh': nn.Linear(input_dim, hidden_dim), 'Wh': nn.Linear(hidden_dim, hidden_dim),
            }) for _ in range(num_layers)
        ])
        self.output_layer = nn.Linear(hidden_dim, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):

        x_normalized = 2.0 * (x - self.lb) / (self.ub - self.lb) - 1.0

        S_state = torch.tanh(self.W1(x_normalized))
        for layer in self.layers:
            Z = torch.sigmoid(layer['Uz'](x_normalized) + layer['Wz'](S_state))
            G = torch.sigmoid(layer['Ug'](x_normalized) + layer['Wg'](S_state))
            R = torch.sigmoid(layer['Ur'](x_normalized) + layer['Wr'](S_state))
            H = torch.tanh(layer['Uh'](x_normalized) + layer['Wh'](S_state * R))
            S_state = (1 - G) * H + Z * S_state
        return self.output_layer(S_state)


import torch
import torch.nn as nn


class ResNet(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, lb, ub, constrain_output=False):

        super(ResNet, self).__init__()

        self.register_buffer('lb', lb)
        self.register_buffer('ub', ub)

        self.input_layer = nn.Linear(input_dim, hidden_dim)

        self.hidden_blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ) for _ in range(num_layers)
        ])

        self.output_layer = nn.Linear(hidden_dim, output_dim)

        self.constrain_output = constrain_output
        if self.constrain_output:
            self.output_head = nn.Sequential(
                self.output_layer,
                nn.Sigmoid()
            )
        else:
            self.output_head = self.output_layer
        # ----------------------------------------------------

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m in [block[-1] for block in self.hidden_blocks]:
                    nn.init.zeros_(m.weight)
                    nn.init.zeros_(m.bias)
                else:
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, x):
        x_normalized = 2.0 * (x - self.lb) / (self.ub - self.lb) - 1.0
        h = self.input_layer(x_normalized)
        for block in self.hidden_blocks:
            h = h + block(h)


        return self.output_head(h)
