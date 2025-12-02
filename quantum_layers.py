"""
Quantum circuit layers for hybrid quantum-classical GANs.

This module provides variational quantum circuits (VQCs) that can be used
as components in hybrid neural network architectures.
"""

import numpy as np
import torch

try:
    import pennylane as qml
    HAS_PENNYLANE = True
except ImportError:
    HAS_PENNYLANE = False


def vqc_output_dim(n_qubits: int = 3) -> int:
    """
    Return the output dimension of the VQC probability vector.

    The VQC as described in the paper emits a probability distribution
    over the computational basis states (length 2**n_qubits). Use
    `vqc_truncated_dim` to compute the post-ancilla truncation size.
    """
    return 2 ** n_qubits


def vqc_truncated_dim(n_qubits: int = 3, n_ancilla: int = 0) -> int:
    """Return the truncated VQC output dimension after removing ancilla elements.

    Paper rule: keep first 2**N - Nancilla elements.
    """
    return max(0, (2 ** n_qubits) - int(n_ancilla))


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

        @qml.qnode(dev, interface='torch')
        def circuit(weights, z_input):
            """VVRQ QNode returning full computational-basis probabilities.

            Args:
                weights: 1D tensor of trainable parameters (length vqc_weight_count).
                z_input: 1D tensor-like of length n_qubits used for angle embedding.

            Returns:
                probs: 1D tensor of length 2**n_qubits with probabilities summing to 1.
            """
            # Input encoding: angle embedding (use entire z_input)
            if z_input is None:
                # deterministic fallback: zeros
                z_tensor = [0.0] * n_qubits
            else:
                # ensure tensor-like sequence of length n_qubits
                z_tensor = z_input

            qml.templates.AngleEmbedding(z_tensor, wires=range(n_qubits), rotation='Y')

            # Variational layers: parameterized single-qubit and entangling rotations
            idx = 0
            for _ in range(n_layers):
                # single-qubit rotations (RY)
                for i in range(n_qubits):
                    qml.RY(weights[idx], wires=i)
                    idx += 1

                # entangling ladder with parameterized RZ after CNOT
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                    qml.RZ(weights[idx], wires=i + 1)
                    idx += 1

            # Return full probability vector over computational basis
            return qml.probs(wires=range(n_qubits))

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

        # Ensure inputs are torch tensors when using the torch interface
        wt = torch.as_tensor(weights[:expected_weights], dtype=torch.float32)
        z_in = None
        if z_input is not None:
            z_in = torch.as_tensor(z_input, dtype=torch.float32)

        result = qnode(wt, z_in)

        # result is a torch tensor (probs); convert to numpy for compatibility
        return result.detach().cpu().numpy()

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


if HAS_PENNYLANE:
    import torch.nn as nn

    class VVRQLayer(nn.Module):
        """PyTorch module wrapping the VVRQ QNode.

        This module owns the trainable weight vector used by the QNode and
        returns the (optionally truncated) probability vector for a given
        batch of z_inputs.
        """

        def __init__(self, n_qubits: int = 3, n_layers: int = 1, n_ancilla: int = 0, batch: bool = False):
            super().__init__()
            self.n_qubits = int(n_qubits)
            self.n_layers = int(n_layers)
            self.n_ancilla = int(n_ancilla)
            # whether to attempt batched QNode execution
            self.batch = bool(batch)

            self._weight_count = vqc_weight_count(self.n_qubits, self.n_layers)
            # register weights as a single parameter vector
            self.weights = nn.Parameter(torch.randn(self._weight_count) * 0.01)

            # create qnode for current geometry
            self._qnode = _create_vvrq_qnode(self.n_qubits, self.n_layers)

        def forward(self, z_inputs: torch.Tensor):
            """Forward pass.

            Args:
                z_inputs: tensor of shape (batch_size, n_qubits) or (n_qubits,) for single sample.

            Returns:
                probs_truncated: torch tensor of shape (batch_size, truncated_dim) or (truncated_dim,)
            """
            single = False
            if z_inputs.dim() == 1:
                z_inputs = z_inputs.unsqueeze(0)
                single = True

            # Try batched execution if requested and supported by the device
            outputs = []
            if self.batch and z_inputs.dim() == 2:
                try:
                    qnode = self._qnode
                    dev = qnode.device
                    # construct tapes for each sample
                    tapes = [qnode.construct((self.weights, z.unsqueeze(0)), {}) for z in z_inputs]
                    results = dev.batch_execute(tapes)
                    # results is a list/array of probability vectors
                    probs_batch = torch.tensor(results).to(dtype=torch.float32)
                    if self.n_ancilla:
                        probs_batch = probs_batch[:, : vqc_truncated_dim(self.n_qubits, self.n_ancilla)]
                    outputs = [p for p in probs_batch]
                except Exception:
                    # fallback to per-sample qnode calls
                    for z in z_inputs:
                        probs = self._qnode(self.weights, z)
                        if self.n_ancilla:
                            probs = probs[: vqc_truncated_dim(self.n_qubits, self.n_ancilla)]
                        probs = probs.to(dtype=torch.float32)
                        outputs.append(probs)
            else:
                for z in z_inputs:
                    probs = self._qnode(self.weights, z)
                    if self.n_ancilla:
                        probs = probs[: vqc_truncated_dim(self.n_qubits, self.n_ancilla)]
                    probs = probs.to(dtype=torch.float32)
                    outputs.append(probs)

            out = torch.stack(outputs, dim=0)
            return out[0] if single else out

    class QDILayer(nn.Module):
        """Quantum Depth-Infused layer using PennyLane QNode.

        Encodes a classical vector into qubit angles via a linear encoder,
        applies repeated variational blocks (n_reps) with n_layers per rep,
        and returns per-qubit PauliZ expectation values as a classical latent.
        """

        def __init__(self, input_dim: int = 64, n_qubits: int = 8, n_reps: int = 8, n_layers: int = 1, batch: bool = False):
            super().__init__()
            self.input_dim = int(input_dim)
            self.n_qubits = int(n_qubits)
            self.n_reps = int(n_reps)
            self.n_layers = int(n_layers)
            self.batch = bool(batch)

            # classical encoder to map input_dim -> n_qubits angles
            self.encoder = nn.Linear(self.input_dim, self.n_qubits)

            # trainable circuit weights count: weights per repetition
            self._weight_count = vqc_weight_count(self.n_qubits, self.n_layers) * self.n_reps
            self.weights = nn.Parameter(torch.randn(self._weight_count) * 0.01)

            # create qnode with desired repetition behavior
            def _create_qdi_qnode(n_qubits=self.n_qubits, n_layers=self.n_layers, n_reps=self.n_reps):
                dev = qml.device('default.qubit', wires=n_qubits)

                @qml.qnode(dev, interface='torch')
                def qdi_node(weights, angles):
                    # angles: 1D tensor of length n_qubits
                    idx = 0
                    for _ in range(n_reps):
                        # encode angles via AngleEmbedding
                        qml.templates.AngleEmbedding(angles, wires=range(n_qubits), rotation='Y')

                        # inner variational layers
                        for _ in range(n_layers):
                            for i in range(n_qubits):
                                qml.RY(weights[idx], wires=i)
                                idx += 1
                            for i in range(n_qubits - 1):
                                qml.CNOT(wires=[i, i + 1])
                                qml.RZ(weights[idx], wires=i + 1)
                                idx += 1

                    # return expectation values per qubit (map to latent z_dim)
                    return tuple(qml.expval(qml.PauliZ(i)) for i in range(n_qubits))

                return qdi_node

            self._qnode = _create_qdi_qnode()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward: x shape (batch, input_dim) -> returns (batch, n_qubits)"""
            single = False
            if x.dim() == 1:
                x = x.unsqueeze(0)
                single = True

            angles = self.encoder(x)  # (batch, n_qubits)

            outputs = []
            # attempt batched execution if requested
            if self.batch and angles.dim() == 2:
                try:
                    qnode = self._qnode
                    dev = qnode.device
                    tapes = [qnode.construct((self.weights, a.unsqueeze(0)), {}) for a in angles]
                    results = dev.batch_execute(tapes)
                    # results expected shape (batch, n_qubits)
                    out_batch = torch.tensor(results).to(dtype=torch.float32)
                    outputs = [o for o in out_batch]
                except Exception:
                    for a in angles:
                        exps = self._qnode(self.weights, a)
                        out = torch.stack(list(exps)).to(dtype=torch.float32)
                        outputs.append(out)
            else:
                for a in angles:
                    exps = self._qnode(self.weights, a)
                    out = torch.stack(list(exps)).to(dtype=torch.float32)
                    outputs.append(out)

            out = torch.stack(outputs, dim=0)
            return out[0] if single else out


else:
    # Fallback pure-PyTorch QDILayer when PennyLane or a compatible SciPy
    # stack is not available. This provides a lightweight, differentiable
    # approximation for unit tests and CI so the rest of the codebase can be
    # exercised without requiring a functional quantum simulator.
    import torch.nn as nn

    class QDILayer(nn.Module):
        """Fallback QDILayer implemented in pure PyTorch.

        The implementation simulates a repeated (n_reps) transformation of
        encoded angles via a weight-derived linear map and a tanh nonlinearity
        to produce outputs in [-1, 1], matching the shape and gradient
        behaviour of the quantum implementation for tests.
        """

        def __init__(self, input_dim: int = 64, n_qubits: int = 8, n_reps: int = 8, n_layers: int = 1, batch: bool = False):
            super().__init__()
            self.input_dim = int(input_dim)
            self.n_qubits = int(n_qubits)
            self.n_reps = int(n_reps)
            self.n_layers = int(n_layers)
            self.batch = bool(batch)

            # classical encoder to map input_dim -> n_qubits angles
            self.encoder = nn.Linear(self.input_dim, self.n_qubits)

            # keep a flat parameter vector to mirror the quantum version
            self._weight_count = vqc_weight_count(self.n_qubits, self.n_layers) * self.n_reps
            # if the weight count is too small to build a square map, allow
            # tiling to produce a square matrix for the simulator
            self.weights = nn.Parameter(torch.randn(self._weight_count) * 0.01)

        def _simulate_one(self, angles: torch.Tensor) -> torch.Tensor:
            """Simulate a single-sample QDI forward using the flat weights.

            Args:
                angles: tensor shape (n_qubits,)
            Returns:
                tensor shape (n_qubits,) with values in [-1, 1]
            """
            # Build a square matrix from weights (n_qubits x n_qubits)
            mat_size = self.n_qubits * self.n_qubits
            if self._weight_count >= mat_size:
                mat = self.weights[:mat_size].view(self.n_qubits, self.n_qubits)
                rem = self.weights[mat_size:]
            else:
                # tile weights to fill matrix
                repeated = self.weights.repeat(int(np.ceil(mat_size / max(1, self._weight_count))))
                mat = repeated[:mat_size].view(self.n_qubits, self.n_qubits)
                rem = repeated[mat_size:]

            # biases from remaining weights (or zeros)
            if rem.numel() >= self.n_qubits:
                bias = rem[:self.n_qubits]
            else:
                bias = torch.zeros(self.n_qubits, device=angles.device, dtype=angles.dtype)

            out = torch.matmul(mat, angles) + bias
            # apply a mild nonlinearity to map to [-1,1]
            out = torch.tanh(out)
            return out

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            single = False
            if x.dim() == 1:
                x = x.unsqueeze(0)
                single = True

            angles = self.encoder(x)  # (batch, n_qubits)

            # vectorized simulation across batch when possible
            if self.batch:
                outs = []
                for a in angles:
                    outs.append(self._simulate_one(a))
                out = torch.stack(outs, dim=0)
            else:
                outs = [self._simulate_one(a) for a in angles]
                out = torch.stack(outs, dim=0)

            return out[0] if single else out



