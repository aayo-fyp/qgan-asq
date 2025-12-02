"""
Cycle components for hybrid quantum-classical GANs.

This module provides classical neural network components that can be used
in cycle-consistent architectures, mirroring the MLP structure used in
the Generator for consistency.
"""

import torch
import torch.nn as nn
from typing import List, Optional

# Optional quantum components (QDILayer may not be available in all environments)
try:
    from quantum_layers import QDILayer
    HAS_QUANTUM = True
except Exception:
    QDILayer = None
    HAS_QUANTUM = False


class ClassicalCycle(nn.Module):
    """
    Classical MLP cycle component that mirrors the Generator's MLP structure.
    
    This can be used as:
    - A classical baseline for comparison with quantum circuits
    - Part of a cycle-consistency architecture (e.g., latent -> features -> latent)
    - A post-processing layer after quantum circuit output
    
    Architecture:
        Input -> [Linear -> Activation -> Dropout] x N -> Output
    
    Args:
        input_dim: Dimension of input features.
        hidden_dims: List of hidden layer dimensions.
        output_dim: Dimension of output features.
        activation: Activation function ('tanh', 'relu', 'leaky_relu', 'sigmoid').
        dropout: Dropout probability (default: 0.0).
        output_activation: Whether to apply activation after final layer (default: False).
    
    Example:
        >>> cycle = ClassicalCycle(input_dim=8, hidden_dims=[64, 32], output_dim=8)
        >>> x = torch.randn(16, 8)  # batch of 16, dim 8
        >>> out = cycle(x)
        >>> out.shape
        torch.Size([16, 8])
    """
    
    ACTIVATIONS = {
        'tanh': nn.Tanh,
        'relu': nn.ReLU,
        'leaky_relu': nn.LeakyReLU,
        'sigmoid': nn.Sigmoid,
        'none': nn.Identity,
    }
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        activation: str = 'tanh',
        dropout: float = 0.0,
        output_activation: bool = False,
    ):
        super(ClassicalCycle, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout_rate = dropout
        
        # Validate activation
        if activation.lower() not in self.ACTIVATIONS:
            raise ValueError(
                f"Unknown activation '{activation}'. "
                f"Choose from: {list(self.ACTIVATIONS.keys())}"
            )
        
        activation_cls = self.ACTIVATIONS[activation.lower()]
        
        # Build layers
        layers = []
        dims = [input_dim] + hidden_dims
        
        for i, (d_in, d_out) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(nn.Linear(d_in, d_out))
            layers.append(activation_cls())
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
        
        self.hidden_layers = nn.Sequential(*layers)
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dims[-1] if hidden_dims else input_dim, output_dim)
        
        # Optional output activation
        self.output_activation = activation_cls() if output_activation else nn.Identity()
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MLP.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim).
            
        Returns:
            Output tensor of shape (batch_size, output_dim).
        """
        if self.hidden_dims:
            x = self.hidden_layers(x)
        x = self.output_layer(x)
        x = self.output_activation(x)
        return x
    
    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"ClassicalCycle(input_dim={self.input_dim}, hidden_dims={self.hidden_dims}, "
            f"output_dim={self.output_dim}, dropout={self.dropout_rate}, params={self.count_parameters()})"
        )


class HQCycle(nn.Module):
    """Hybrid Quantum Cycle: Classical encoder -> QDI quantum block -> optional mapper.

    This composes a `ClassicalCycle` that reduces a flattened graph/features tensor
    to an intermediate classical vector (default 64), then feeds that vector into
    a `QDILayer` which encodes into qubits using repeated encoding layers and
    returns a per-qubit expectation vector. An optional linear mapper projects
    the quantum output to the desired latent `z_dim` for the Generator.
    """

    def __init__(
        self,
        input_dim: int,
        intermediate_dim: int = 64,
        z_dim: int = 8,
        classical_hidden: Optional[List[int]] = None,
        qdi_kwargs: Optional[dict] = None,
    ):
        super(HQCycle, self).__init__()

        if not HAS_QUANTUM or QDILayer is None:
            raise RuntimeError("QDILayer (pennylane) not available in this environment")

        self.input_dim = int(input_dim)
        self.intermediate_dim = int(intermediate_dim)
        self.z_dim = int(z_dim)

        classical_hidden = classical_hidden or [128, 64]
        self.classical_encoder = ClassicalCycle(
            input_dim=self.input_dim,
            hidden_dims=classical_hidden,
            output_dim=self.intermediate_dim,
            activation='tanh',
            dropout=0.0,
            output_activation=False,
        )

        qdi_kwargs = qdi_kwargs or {}
        # Ensure QDILayer n_qubits matches desired z_dim by default
        n_qubits = qdi_kwargs.get('n_qubits', self.z_dim)
        qdi_call_kwargs = dict(qdi_kwargs)
        qdi_call_kwargs.pop('n_qubits', None)
        self.qdi = QDILayer(input_dim=self.intermediate_dim, n_qubits=n_qubits, **qdi_call_kwargs)

        # If quantum output dim differs from requested z_dim, add a mapper
        if self.qdi.n_qubits != self.z_dim:
            self.post_mapper = nn.Linear(self.qdi.n_qubits, self.z_dim)
        else:
            self.post_mapper = None

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: x -> classical encoder -> qdi -> optional mapper -> z

        Args:
            x: Tensor of shape (batch, input_dim)

        Returns:
            z: Tensor of shape (batch, z_dim)
        """
        # classical reduction
        hidden = self.classical_encoder(x)

        # quantum block: returns (batch, n_qubits)
        q_out = self.qdi(hidden)

        if self.post_mapper is not None:
            z = self.post_mapper(q_out)
        else:
            z = q_out

        return z

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"HQCycle(input_dim={self.input_dim}, intermediate_dim={self.intermediate_dim}, "
            f"z_dim={self.z_dim}, params={self.count_parameters()})"
        )


class LatentMapper(nn.Module):
    """
    Maps between different latent space dimensions.
    
    Useful for connecting quantum circuit outputs (e.g., 8 dims from 8 qubits)
    to classical network inputs that may expect different dimensions.
    
    Args:
        input_dim: Input dimension (e.g., quantum circuit output).
        output_dim: Output dimension (e.g., generator input).
        hidden_dim: Optional hidden layer dimension for non-linear mapping.
        activation: Activation function if hidden_dim is used.
    
    Example:
        >>> mapper = LatentMapper(input_dim=8, output_dim=128, hidden_dim=64)
        >>> z_quantum = torch.randn(16, 8)  # 8 qubits output
        >>> z_mapped = mapper(z_quantum)
        >>> z_mapped.shape
        torch.Size([16, 128])
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: Optional[int] = None,
        activation: str = 'tanh',
    ):
        super(LatentMapper, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        if hidden_dim is not None:
            activation_cls = ClassicalCycle.ACTIVATIONS.get(activation.lower(), nn.Tanh)
            self.layers = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                activation_cls(),
                nn.Linear(hidden_dim, output_dim),
            )
        else:
            # Simple linear projection
            self.layers = nn.Linear(input_dim, output_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.layers(x)
    
    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

