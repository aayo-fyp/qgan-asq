"""
Quantum circuit layers for hybrid quantum-classical GANs.

This module provides variational quantum circuits (VQCs) that can be used
as components in hybrid neural network architectures. The implementation
follows the VVRQ (Variational Very-Random Quantum) architecture from the
hybrid quantum GAN paper.

Key features:
- Outputs probability distribution over 2^n computational basis states
- PyTorch-compatible interface for gradient-based training
- Support for truncation to remove ancilla qubits
"""

import torch
import torch.nn as nn

try:
    import pennylane as qml
    HAS_PENNYLANE = True
except ImportError:
    HAS_PENNYLANE = False


# =============================================================================
# Helper Functions
# =============================================================================

def vqc_output_dim(n_qubits: int = 3) -> int:
    """
    Return the output dimension of the VVRQ circuit (probability distribution).
    
    The VVRQ circuit outputs probabilities over all 2^n computational basis states.
    
    Args:
        n_qubits: Number of qubits in the circuit.
        
    Returns:
        Output dimension (2^n_qubits).
        
    Example:
        >>> vqc_output_dim(3)
        8
        >>> vqc_output_dim(4)
        16
    """
    return 2 ** n_qubits


def vqc_truncated_dim(n_qubits: int, n_ancilla: int = 0) -> int:
    """
    Return the truncated output dimension after removing ancilla states.
    
    Per the paper, the VQC output is truncated to keep the first
    (2^n - n_ancilla) elements of the probability distribution.
    
    Args:
        n_qubits: Number of qubits in the circuit.
        n_ancilla: Number of ancilla states to remove (default: 0).
        
    Returns:
        Truncated output dimension (2^n_qubits - n_ancilla).
        
    Example:
        >>> vqc_truncated_dim(3, n_ancilla=0)
        8
        >>> vqc_truncated_dim(3, n_ancilla=3)
        5
    """
    return 2 ** n_qubits - n_ancilla


def vqc_weight_count(n_qubits: int = 3, n_layers: int = 1) -> int:
    """
    Return the number of trainable weights required for the VVRQ circuit.
    
    Each layer has:
    - n_qubits RY rotation parameters
    - (n_qubits - 1) RZ rotation parameters for entangling gates
    
    Args:
        n_qubits: Number of qubits in the circuit.
        n_layers: Number of variational layers.
        
    Returns:
        Total number of trainable parameters.
        
    Example:
        >>> vqc_weight_count(3, 1)  # 3 RY + 2 RZ = 5
        5
        >>> vqc_weight_count(4, 2)  # 2 * (4 RY + 3 RZ) = 14
        14
    """
    return n_layers * (n_qubits + (n_qubits - 1))


def truncate_probs(probs: torch.Tensor, n_ancilla: int = 0) -> torch.Tensor:
    """
    Truncate probability distribution by removing ancilla states.
    
    Per the paper, keep the first (2^n - n_ancilla) elements.
    
    Args:
        probs: Probability tensor of shape (..., 2^n_qubits).
        n_ancilla: Number of ancilla states to remove from the end.
        
    Returns:
        Truncated tensor of shape (..., 2^n_qubits - n_ancilla).
        
    Example:
        >>> probs = torch.tensor([0.1, 0.2, 0.3, 0.15, 0.1, 0.05, 0.05, 0.05])
        >>> truncate_probs(probs, n_ancilla=3)
        tensor([0.1000, 0.2000, 0.3000, 0.1500, 0.1000])
    """
    if n_ancilla == 0:
        return probs
    if n_ancilla >= probs.shape[-1]:
        raise ValueError(
            f"n_ancilla ({n_ancilla}) must be less than output dim ({probs.shape[-1]})"
        )
    return probs[..., :-n_ancilla]


# =============================================================================
# Quantum Circuit Implementation
# =============================================================================

if HAS_PENNYLANE:
    
    def _create_vvrq_qnode(n_qubits: int = 3, n_layers: int = 1, device_name: str = 'default.qubit'):
        """
        Create a PennyLane QNode for the VVRQ circuit with PyTorch interface.
        
        Architecture:
        - Input encoding: RY and RZ rotations based on z_input
        - Variational layers: RY rotations + CNOT ladder with RZ rotations
        - Measurement: Probability distribution over all computational basis states
        
        Args:
            n_qubits: Number of qubits.
            n_layers: Number of variational layers.
            device_name: PennyLane device name (default: 'default.qubit').
            
        Returns:
            A PennyLane QNode function compatible with PyTorch.
        """
        dev = qml.device(device_name, wires=n_qubits)
        
        @qml.qnode(dev, interface='torch', diff_method='backprop')
        def circuit(weights, z_input):
            """
            Execute the VVRQ circuit.
            
            Args:
                weights: 1D tensor of trainable parameters.
                z_input: 1D tensor of shape (2,) for input encoding.
                
            Returns:
                Probability distribution tensor of shape (2^n_qubits,).
            """
            z1 = z_input[0]
            z2 = z_input[1]
            
            # Input encoding: apply RY and RZ based on z_input
            # Use arcsin for angle mapping (maps [-1, 1] to [-pi/2, pi/2])
            for i in range(n_qubits):
                qml.RY(torch.arcsin(z1), wires=i)
                qml.RZ(torch.arcsin(z2), wires=i)
            
            # Variational layers
            idx = 0
            for _ in range(n_layers):
                # Single-qubit RY rotations
                for i in range(n_qubits):
                    qml.RY(weights[idx], wires=i)
                    idx += 1
                
                # Entangling gates: CNOT ladder with parameterized RZ
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                    qml.RZ(weights[idx], wires=i + 1)
                    qml.CNOT(wires=[i, i + 1])
                    idx += 1
            
            # Return probability distribution over all basis states
            return qml.probs(wires=range(n_qubits))
        
        return circuit

    class VVRQLayer(nn.Module):
        """
        PyTorch nn.Module wrapper for the VVRQ quantum circuit.
        
        This layer integrates seamlessly with PyTorch models and optimizers.
        It outputs a probability distribution over 2^n computational basis states.
        
        Args:
            n_qubits: Number of qubits (default: 3).
            n_layers: Number of variational layers (default: 1).
            n_ancilla: Number of ancilla states to truncate (default: 0).
            device_name: PennyLane device name (default: 'default.qubit').
            
        Attributes:
            weights: Trainable parameters (nn.Parameter).
            n_qubits: Number of qubits.
            n_layers: Number of variational layers.
            n_ancilla: Number of ancilla states to truncate.
            
        Example:
            >>> layer = VVRQLayer(n_qubits=3, n_layers=1)
            >>> z_input = torch.tensor([0.5, -0.3])
            >>> output = layer(z_input)
            >>> output.shape
            torch.Size([8])
            >>> # With truncation
            >>> layer = VVRQLayer(n_qubits=3, n_layers=1, n_ancilla=3)
            >>> output = layer(z_input)
            >>> output.shape
            torch.Size([5])
        """
        
        def __init__(
            self,
            n_qubits: int = 3,
            n_layers: int = 1,
            n_ancilla: int = 0,
            device_name: str = 'default.qubit'
        ):
            super().__init__()
            self.n_qubits = n_qubits
            self.n_layers = n_layers
            self.n_ancilla = n_ancilla
            self.device_name = device_name
            
            # Initialize trainable weights
            n_weights = vqc_weight_count(n_qubits, n_layers)
            self.weights = nn.Parameter(torch.randn(n_weights) * 0.1)
            
            # Create the quantum circuit
            self._qnode = _create_vvrq_qnode(n_qubits, n_layers, device_name)
        
        def forward(self, z_input: torch.Tensor) -> torch.Tensor:
            """
            Forward pass through the quantum circuit.
            
            Args:
                z_input: Input tensor of shape (2,) or (batch_size, 2).
                         Values should be in [-1, 1] for proper encoding.
                         
            Returns:
                Probability distribution tensor of shape (output_dim,) or (batch_size, output_dim),
                where output_dim = 2^n_qubits - n_ancilla.
            """
            # Handle batched input
            if z_input.dim() == 1:
                probs = self._qnode(self.weights, z_input)
                probs = truncate_probs(probs, self.n_ancilla)
                return probs.float()  # Ensure float32 for PyTorch compatibility
            else:
                # Process batch sequentially (PennyLane batching can be added later)
                batch_probs = []
                for i in range(z_input.shape[0]):
                    probs = self._qnode(self.weights, z_input[i])
                    batch_probs.append(truncate_probs(probs, self.n_ancilla))
                return torch.stack(batch_probs).float()  # Ensure float32
        
        def output_dim(self) -> int:
            """Return the output dimension after truncation."""
            return vqc_truncated_dim(self.n_qubits, self.n_ancilla)
        
        def extra_repr(self) -> str:
            return (
                f'n_qubits={self.n_qubits}, n_layers={self.n_layers}, '
                f'n_ancilla={self.n_ancilla}, output_dim={self.output_dim()}'
            )

    def vvrq_vqc(
        weights: torch.Tensor,
        z_input: torch.Tensor,
        n_qubits: int = 3,
        n_layers: int = 1,
        n_ancilla: int = 0
    ) -> torch.Tensor:
        """
        Execute the VVRQ variational quantum circuit (functional interface).
        
        This is a functional interface for using the quantum circuit. For integration
        with PyTorch models, prefer using the VVRQLayer nn.Module.
        
        Args:
            weights: 1D tensor with trainable parameters.
                     Required length: vqc_weight_count(n_qubits, n_layers)
            z_input: Input tensor of shape (2,) with values in [-1, 1].
            n_qubits: Number of qubits (default: 3).
            n_layers: Number of variational layers (default: 1).
            n_ancilla: Number of ancilla states to truncate (default: 0).
        
        Returns:
            Probability distribution tensor of shape (2^n_qubits - n_ancilla,).
        
        Example:
            >>> weights = torch.randn(5) * 0.1  # 5 params for 3 qubits, 1 layer
            >>> z_input = torch.tensor([0.5, -0.3])
            >>> output = vvrq_vqc(weights, z_input, n_qubits=3, n_layers=1)
            >>> output.shape
            torch.Size([8])
        """
        expected_weights = vqc_weight_count(n_qubits, n_layers)
        if weights.shape[0] < expected_weights:
            raise ValueError(
                f"Expected at least {expected_weights} weights for {n_qubits} qubits "
                f"and {n_layers} layers, got {weights.shape[0]}"
            )
        
        # Ensure inputs are tensors
        if not isinstance(weights, torch.Tensor):
            weights = torch.tensor(weights, dtype=torch.float32)
        if not isinstance(z_input, torch.Tensor):
            z_input = torch.tensor(z_input, dtype=torch.float32)
        
        qnode = _create_vvrq_qnode(n_qubits, n_layers)
        probs = qnode(weights[:expected_weights], z_input)
        
        return truncate_probs(probs, n_ancilla).float()  # Ensure float32

else:
    # Fallback when PennyLane is not installed
    
    class VVRQLayer(nn.Module):
        """Placeholder VVRQLayer when PennyLane is not installed."""
        
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "PennyLane is not installed. The VVRQLayer requires PennyLane for "
                "quantum circuit simulation. Install it with: pip install pennylane\n"
                "For GPU acceleration, also install: pip install pennylane-lightning-gpu"
            )
    
    def vvrq_vqc(*args, **kwargs):
        """Placeholder when PennyLane is not installed."""
        raise RuntimeError(
            "PennyLane is not installed. The vvrq_vqc function requires PennyLane for "
            "quantum circuit simulation. Install it with: pip install pennylane\n"
            "For GPU acceleration, also install: pip install pennylane-lightning-gpu"
        )
