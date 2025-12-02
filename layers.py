import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):

    def __init__(self, in_features, out_feature_list, b_dim, dropout):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_feature_list = out_feature_list

        self.linear1 = nn.Linear(in_features, out_feature_list[0])
        self.linear2 = nn.Linear(out_feature_list[0], out_feature_list[1])

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, adj, activation=None):
        # input (node features): (batch_size, num_atoms, atom_feature_dim)
        # adj (adjacency matrices): (batch_size, num_bond_types, num_atoms, num_atoms)
        # input : 16x9x9
        # adj : 16x4x9x9
        
        # === First Convolutional Hop ===
        # 1. Transform node features and prepare for bond-specific aggregation
        hidden = torch.stack([self.linear1(input) for _ in range(adj.size(1))], 1)
        # 2. Aggregate neighbor features for each bond type
        hidden = torch.einsum('bijk,bikl->bijl', (adj, hidden))
        # 3. Sum aggregations over all bond types and add a skip connection
        hidden = torch.sum(hidden, 1) + self.linear1(input)
        # 5. Apply activation and dropout
        hidden = activation(hidden) if activation is not None else hidden
        hidden = self.dropout(hidden)

        # === Second Convolutional Hop ===
        # Same process is repeated with the next linear layer
        output = torch.stack([self.linear2(hidden) for _ in range(adj.size(1))], 1)
        output = torch.einsum('bijk,bikl->bijl', (adj, output))
        output = torch.sum(output, 1) + self.linear2(hidden)
        output = activation(output) if activation is not None else output
        output = self.dropout(output)

        return output


class GraphAggregation(Module):

    def __init__(self, in_features, out_features, b_dim, dropout):
        super(GraphAggregation, self).__init__()
        self.sigmoid_linear = nn.Sequential(nn.Linear(in_features+b_dim, out_features),
                                            nn.Sigmoid())
        self.tanh_linear = nn.Sequential(nn.Linear(in_features+b_dim, out_features),
                                         nn.Tanh())
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, activation):
        # input (all node features): (batch_size, num_atoms, feature_dim)

        # 1. Calculate a "gate" value for each atom's features
        i = self.sigmoid_linear(input)
        # 2. Calculate a "candidate" value for each atom's features
        j = self.tanh_linear(input)
        # 3. Apply the gate to the candidate features and sum gated features across all atoms
        output = torch.sum(torch.mul(i,j), 1)
        # 4. Apply final activation and dropout
        output = activation(output) if activation is not None\
                 else output
        output = self.dropout(output)

        return output
