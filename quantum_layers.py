"""
Quantum circuit layers for hybrid quantum-classical GANs.

This module provides variational quantum circuits (VQCs) that can be used
as components in hybrid neural network architectures.
"""

import numpy as np

try:
    import pennylane as qml
    HAS_PENNYLANE = True
except ImportError:
    HAS_PENNYLANE = False


def vqc_output_dim(n_qubits: int = 3) -> int:
    """
    Return the output dimension of the VVRQ circuit.
    
    The VVRQ circuit outputs one expectation value per qubit.
    
    Args:
        n_qubits: Number of qubits in the circuit.
        
    Returns:
        Output dimension (equal to n_qubits).
    """
    return n_qubits


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
    """
    return n_layers * (n_qubits + (n_qubits - 1))


if HAS_PENNYLANE:
    def _create_vvrq_qnode(n_qubits: int = 3, n_layers: int = 1):
        """
        Create a PennyLane QNode for the VVRQ (Variational Very-Random Quantum) circuit.
        
        Architecture:
        - Input encoding: RY and RZ rotations with random noise
        - Variational layers: RY rotations + CNOT ladder with RZ rotations
        - Measurement: PauliZ expectation values on all qubits
        
        Args:
            n_qubits: Number of qubits.
            n_layers: Number of variational layers.
            
        Returns:
            A PennyLane QNode function.
        """
        dev = qml.device('default.qubit', wires=n_qubits)
        
        @qml.qnode(dev, interface='autograd')
        def circuit(weights, z_input=None):
            """
            Execute the VVRQ circuit.
            
            Args:
                weights: 1D array of trainable parameters.
                z_input: Optional 2D input (z1, z2) for encoding. If None, random values used.
                
            Returns:
                Tuple of expectation values.
            """
            # Input encoding with noise
            if z_input is None:
                z1 = np.random.uniform(-1, 1)
                z2 = np.random.uniform(-1, 1)
            else:
                z1, z2 = z_input[0], z_input[1]
            
            for i in range(n_qubits):
                qml.RY(np.arcsin(z1), wires=i)
                qml.RZ(np.arcsin(z2), wires=i)
            
            # Variational layers
            idx = 0
            for _ in range(n_layers):
                # Single-qubit rotations
                for i in range(n_qubits):
                    qml.RY(weights[idx], wires=i)
                    idx += 1
                
                # Entangling gates with parameterized rotations
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                    qml.RZ(weights[idx], wires=i + 1)
                    qml.CNOT(wires=[i, i + 1])
                    idx += 1
            
            # Measure all qubits
            return tuple(qml.expval(qml.PauliZ(i)) for i in range(n_qubits))
        
        return circuit

    def vvrq_vqc(weights, n_qubits: int = 3, n_layers: int = 1, z_input=None):
        """
        Execute the VVRQ variational quantum circuit.
        
        This is the main interface for using the quantum circuit in hybrid models.
        The circuit encodes random noise and applies trainable variational layers,
        outputting expectation values that can be used as latent representations.
        
        Args:
            weights: 1D numpy array or tensor with trainable parameters.
                     Required length: vqc_weight_count(n_qubits, n_layers)
            n_qubits: Number of qubits (default: 3).
            n_layers: Number of variational layers (default: 1).
            z_input: Optional tuple (z1, z2) for deterministic input encoding.
                     If None, random values in [-1, 1] are used.
        
        Returns:
            numpy array of shape (n_qubits,) with expectation values in [-1, 1].
        
        Example:
            >>> weights = np.random.rand(5) * 0.1  # 5 params for 3 qubits, 1 layer
            >>> output = vvrq_vqc(weights, n_qubits=3, n_layers=1)
            >>> output.shape
            (3,)
        """
        expected_weights = vqc_weight_count(n_qubits, n_layers)
        if len(weights) < expected_weights:
            raise ValueError(
                f"Expected at least {expected_weights} weights for {n_qubits} qubits "
                f"and {n_layers} layers, got {len(weights)}"
            )
        
        qnode = _create_vvrq_qnode(n_qubits, n_layers)
        result = qnode(weights[:expected_weights], z_input=z_input)
        
        # Convert tuple of tensors to numpy array
        return np.array([float(r) for r in result])

else:
    def vvrq_vqc(weights, n_qubits: int = 3, n_layers: int = 1, z_input=None):
        """
        Placeholder when PennyLane is not installed.
        
        Raises:
            RuntimeError: Always raised indicating PennyLane is required.
        """
        raise RuntimeError(
            "PennyLane is not installed. Install it with: pip install pennylane"
        )

