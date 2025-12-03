import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers import GraphConvolution, GraphAggregation

# optional quantum layer (soft dependency)
try:
    from quantum_layers import VVRQLayer, vqc_truncated_dim
    HAS_QUANTUM = True
except Exception:
    VVRQLayer = None
    vqc_truncated_dim = None
    HAS_QUANTUM = False


class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        return x + self.main(x)


class Generator(nn.Module):
    """Generator network."""
    def __init__(self, conv_dims, z_dim, vertexes, edges, nodes, dropout,
                 quantum: bool = False, vqc_kwargs: dict = None):
        super(Generator, self).__init__()

        self.vertexes = vertexes
        self.edges = edges
        self.nodes = nodes

        self.quantum = bool(quantum) and HAS_QUANTUM
        self.vqc_mapper = None
        if self.quantum:
            # vqc_kwargs should contain n_qubits, n_layers, n_ancilla
            vqc_kwargs = vqc_kwargs or {}
            n_qubits = int(vqc_kwargs.get('n_qubits', 3))
            n_layers = int(vqc_kwargs.get('n_layers', 1))
            n_ancilla = int(vqc_kwargs.get('n_ancilla', 0))
            # create VVRQ layer
            self.vqc = VVRQLayer(n_qubits=n_qubits, n_layers=n_layers, n_ancilla=n_ancilla)
            # linear mapper from truncated vqc dim -> expected z_dim
            truncated = vqc_truncated_dim(n_qubits, n_ancilla)
            self.vqc_mapper = nn.Linear(truncated, z_dim)

        layers = []
        for c0, c1 in zip([z_dim] + conv_dims[:-1], conv_dims):
            layers.append(nn.Linear(c0, c1))
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(p=dropout, inplace=True))
        self.layers = nn.Sequential(*layers)

        self.edges_layer = nn.Linear(conv_dims[-1], edges * vertexes * vertexes)
        self.nodes_layer = nn.Linear(conv_dims[-1], vertexes * nodes)
        self.dropoout = nn.Dropout(p=dropout)

    def forward(self, x):
        # if quantum mode is enabled, x should be the noise vector consumed by the VQC
        if self.quantum:
            # pass through quantum layer -> mapper -> acts as latent z
            z_q = self.vqc(x)
            z = self.vqc_mapper(z_q)
            output = self.layers(z)
        else:
            output = self.layers(x)

        edges_logits = self.edges_layer(output) \
                       .view(-1, self.edges, self.vertexes, self.vertexes)
        edges_logits = (edges_logits + edges_logits.permute(0, 1, 3, 2)) / 2
        edges_logits = self.dropoout(edges_logits.permute(0, 2, 3, 1))

        nodes_logits = self.nodes_layer(output)
        nodes_logits = self.dropoout(nodes_logits.view(-1, self.vertexes, self.nodes))

        return edges_logits, nodes_logits


class Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""
    def __init__(self, conv_dim, m_dim, b_dim, dropout):
        """
        conv_dim: tuple (graph_conv_dim, aux_dim, linear_dim).
            graph_conv_dim is a list like [128, 64] and graph_conv_dim[-1] is used.
        m_dim: node feature dim (m_dim)
        b_dim: auxiliary dim appended to in_features when building GraphAggregation
        dropout: dropout rate
        """
        super(Discriminator, self).__init__()

        graph_conv_dim, aux_dim, linear_dim = conv_dim
        self.graph_conv_dim = graph_conv_dim
        self.aux_dim = aux_dim
        self.linear_dim = linear_dim
        self.init_b_dim = b_dim  # keep original value for reference
        self.dropout = dropout

        # discriminator
        self.gcn_layer = GraphConvolution(m_dim, graph_conv_dim, b_dim, dropout)
        # build aggregation with the provided b_dim initially
        self.agg_layer = GraphAggregation(graph_conv_dim[-1], aux_dim, b_dim, dropout)

        # multi dense layer
        layers = []
        for c0, c1 in zip([aux_dim] + linear_dim[:-1], linear_dim):
            layers.append(nn.Linear(c0, c1))
            layers.append(nn.Dropout(dropout))
        self.linear_layer = nn.Sequential(*layers)

        self.output_layer = nn.Linear(linear_dim[-1], 1)

    def _rebuild_agg_layer(self, annotations):
        """
        Rebuild self.agg_layer to match the runtime annotations size if the
        expected input size differs from actual annotations.shape[-1].
        """
        # annotations shape: (batch, vertexes, annotation_dim)
        annotation_dim = annotations.shape[-1]
        expected_in = self.graph_conv_dim[-1]  # h dim
        # deduce runtime b_dim:
        runtime_b_dim = annotation_dim - expected_in
        if runtime_b_dim <= 0:
            raise RuntimeError(f"Runtime computed b_dim={runtime_b_dim} is invalid (<=0). "
                               f"annotation_dim={annotation_dim}, expected_in={expected_in}")
        # recreate the aggregation layer with the runtime b_dim
        self.agg_layer = GraphAggregation(expected_in, self.aux_dim, runtime_b_dim, self.dropout)
        # (optional) log/debug:
        # print(f"[models.Discriminator] Rebuilt agg_layer with runtime_b_dim={runtime_b_dim}, annotation_dim={annotation_dim}")

    def forward(self, adj, hidden, node, activatation=None):
        # adj expected in shape (batch, ???, vertexes, vertexes+1) per calling code;
        # original code took adj[:,:,:,1:].permute(0,3,1,2)
        adj = adj[:, :, :, 1:].permute(0, 3, 1, 2)

        # annotations: if hidden present, concat(hidden, node), else node
        annotations1 = torch.cat((hidden, node), -1) if hidden is not None else node

        # pass through GCN
        h = self.gcn_layer(annotations1, adj)
        # create annotations2: concat(h, hidden, node) (or h,node if no hidden)
        annotations2 = torch.cat((h, hidden, node) if hidden is not None else (h, node), -1)

        # Defensive: ensure agg_layer was created with matching in_features + b_dim.
        # The first Linear in agg_layer.sigmoid_linear is the Linear module we want to match.
        try:
            expected_linear_in = self.agg_layer.sigmoid_linear[0].in_features
        except Exception:
            expected_linear_in = None

        if expected_linear_in is None or expected_linear_in != annotations2.shape[-1]:
            # rebuild to match runtime shape
            self._rebuild_agg_layer(annotations2)

        # pass through aggregation and linear layers
        h_agg = self.agg_layer(annotations2, torch.tanh)
        h_lin = self.linear_layer(h_agg)

        output = self.output_layer(h_lin)
        output = activatation(output) if activatation is not None else output

        return output, h_lin