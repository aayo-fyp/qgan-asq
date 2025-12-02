"""
Unit tests for quantum_layers module.

Tests cover:
- Helper functions (vqc_output_dim, vqc_truncated_dim, vqc_weight_count, truncate_probs)
- VVRQ quantum circuit (functional interface)
- VVRQLayer nn.Module wrapper
- Gradient flow verification
"""

import numpy as np
import pytest
import torch

import sys
sys.path.insert(0, '..')

import quantum_layers as ql


class TestVQCHelpers:
    """Tests for helper functions."""
    
    def test_vqc_output_dim_default(self):
        """Output dimension is 2^n_qubits."""
        assert ql.vqc_output_dim() == 8  # 2^3 = 8
        assert ql.vqc_output_dim(n_qubits=3) == 8
    
    def test_vqc_output_dim_various(self):
        """Output dimension scales exponentially with qubits."""
        assert ql.vqc_output_dim(n_qubits=1) == 2
        assert ql.vqc_output_dim(n_qubits=2) == 4
        assert ql.vqc_output_dim(n_qubits=4) == 16
        assert ql.vqc_output_dim(n_qubits=5) == 32
    
    def test_vqc_truncated_dim_no_ancilla(self):
        """Without ancilla, truncated dim equals full dim."""
        assert ql.vqc_truncated_dim(3, n_ancilla=0) == 8
        assert ql.vqc_truncated_dim(4, n_ancilla=0) == 16
    
    def test_vqc_truncated_dim_with_ancilla(self):
        """Truncated dim is 2^n - n_ancilla."""
        assert ql.vqc_truncated_dim(3, n_ancilla=3) == 5
        assert ql.vqc_truncated_dim(4, n_ancilla=8) == 8
        assert ql.vqc_truncated_dim(3, n_ancilla=1) == 7
    
    def test_vqc_weight_count_default(self):
        """Weight count for 3 qubits, 1 layer = 3 + 2 = 5."""
        assert ql.vqc_weight_count() == 5
        assert ql.vqc_weight_count(n_qubits=3, n_layers=1) == 5
    
    def test_vqc_weight_count_various(self):
        """Weight count scales correctly."""
        # 1 qubit, 1 layer: 1 RY + 0 RZ = 1
        assert ql.vqc_weight_count(n_qubits=1, n_layers=1) == 1
        # 4 qubits, 1 layer: 4 RY + 3 RZ = 7
        assert ql.vqc_weight_count(n_qubits=4, n_layers=1) == 7
        # 3 qubits, 2 layers: 2 * (3 + 2) = 10
        assert ql.vqc_weight_count(n_qubits=3, n_layers=2) == 10
        # 8 qubits, 1 layer: 8 + 7 = 15
        assert ql.vqc_weight_count(n_qubits=8, n_layers=1) == 15


class TestTruncateProbs:
    """Tests for truncate_probs function."""
    
    def test_truncate_no_ancilla(self):
        """No truncation when n_ancilla=0."""
        probs = torch.tensor([0.125] * 8)
        result = ql.truncate_probs(probs, n_ancilla=0)
        assert result.shape == (8,)
        torch.testing.assert_close(result, probs)
    
    def test_truncate_with_ancilla(self):
        """Truncates last n_ancilla elements."""
        probs = torch.tensor([0.1, 0.2, 0.3, 0.15, 0.1, 0.05, 0.05, 0.05])
        result = ql.truncate_probs(probs, n_ancilla=3)
        expected = torch.tensor([0.1, 0.2, 0.3, 0.15, 0.1])
        assert result.shape == (5,)
        torch.testing.assert_close(result, expected)
    
    def test_truncate_batched(self):
        """Works with batched input."""
        probs = torch.rand(4, 8)
        result = ql.truncate_probs(probs, n_ancilla=2)
        assert result.shape == (4, 6)
    
    def test_truncate_invalid_ancilla_raises(self):
        """Raises error if n_ancilla >= output dim."""
        probs = torch.tensor([0.5, 0.5])
        with pytest.raises(ValueError, match="n_ancilla"):
            ql.truncate_probs(probs, n_ancilla=2)


class TestVVRQCircuit:
    """Tests for the VVRQ quantum circuit (functional interface)."""
    
    @pytest.fixture(autouse=True)
    def check_pennylane(self):
        """Skip tests if PennyLane is not installed."""
        if not ql.HAS_PENNYLANE:
            pytest.skip("PennyLane not installed")
    
    def test_vvrq_output_shape_default(self):
        """Default circuit outputs 2^3 = 8 probabilities."""
        weights = torch.randn(5) * 0.1
        z_input = torch.tensor([0.5, -0.3])
        output = ql.vvrq_vqc(weights, z_input)
        assert output.shape == (8,)
    
    def test_vvrq_output_shape_various(self):
        """Output shape is 2^n_qubits."""
        # 4 qubits -> 16 outputs
        weights = torch.randn(7) * 0.1
        z_input = torch.tensor([0.5, -0.3])
        output = ql.vvrq_vqc(weights, z_input, n_qubits=4)
        assert output.shape == (16,)
        
        # 2 qubits -> 4 outputs
        weights = torch.randn(3) * 0.1
        output = ql.vvrq_vqc(weights, z_input, n_qubits=2)
        assert output.shape == (4,)
    
    def test_vvrq_output_is_probability(self):
        """Output is a valid probability distribution."""
        weights = torch.randn(5) * 0.1
        z_input = torch.tensor([0.5, -0.3])
        output = ql.vvrq_vqc(weights, z_input, n_qubits=3)
        
        # All values non-negative
        assert (output >= 0).all(), f"Negative probabilities: {output}"
        # Sum to 1
        assert torch.isclose(output.sum(), torch.tensor(1.0), atol=1e-5), f"Sum != 1: {output.sum()}"
    
    def test_vvrq_deterministic(self):
        """Same inputs produce same output."""
        weights = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
        z_input = torch.tensor([0.5, -0.3])
        
        output1 = ql.vvrq_vqc(weights, z_input)
        output2 = ql.vvrq_vqc(weights, z_input)
        
        torch.testing.assert_close(output1, output2)
    
    def test_vvrq_different_weights_different_output(self):
        """Different weights produce different outputs."""
        z_input = torch.tensor([0.5, -0.3])
        
        weights1 = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
        weights2 = torch.tensor([0.5, 0.4, 0.3, 0.2, 0.1])
        
        output1 = ql.vvrq_vqc(weights1, z_input)
        output2 = ql.vvrq_vqc(weights2, z_input)
        
        assert not torch.allclose(output1, output2), "Different weights should produce different outputs"
    
    def test_vvrq_insufficient_weights_raises(self):
        """Raises ValueError if not enough weights provided."""
        weights = torch.tensor([0.1, 0.2])  # Only 2, need 5 for 3 qubits
        z_input = torch.tensor([0.5, -0.3])
        
        with pytest.raises(ValueError, match="Expected at least 5 weights"):
            ql.vvrq_vqc(weights, z_input, n_qubits=3)
    
    def test_vvrq_with_truncation(self):
        """Truncation reduces output size."""
        weights = torch.randn(5) * 0.1
        z_input = torch.tensor([0.5, -0.3])
        
        output = ql.vvrq_vqc(weights, z_input, n_qubits=3, n_ancilla=3)
        assert output.shape == (5,)  # 8 - 3 = 5
    
    def test_vvrq_multiple_layers(self):
        """Circuit works with multiple layers."""
        # 3 qubits, 2 layers = 10 weights
        weights = torch.randn(10) * 0.1
        z_input = torch.tensor([0.5, -0.3])
        output = ql.vvrq_vqc(weights, z_input, n_qubits=3, n_layers=2)
        
        assert output.shape == (8,)
        assert (output >= 0).all()
        assert torch.isclose(output.sum(), torch.tensor(1.0), atol=1e-5)


class TestVVRQLayer:
    """Tests for the VVRQLayer nn.Module."""
    
    @pytest.fixture(autouse=True)
    def check_pennylane(self):
        """Skip tests if PennyLane is not installed."""
        if not ql.HAS_PENNYLANE:
            pytest.skip("PennyLane not installed")
    
    def test_layer_creation(self):
        """Layer can be created with default params."""
        layer = ql.VVRQLayer()
        assert layer.n_qubits == 3
        assert layer.n_layers == 1
        assert layer.n_ancilla == 0
        assert layer.output_dim() == 8
    
    def test_layer_custom_params(self):
        """Layer respects custom parameters."""
        layer = ql.VVRQLayer(n_qubits=4, n_layers=2, n_ancilla=5)
        assert layer.n_qubits == 4
        assert layer.n_layers == 2
        assert layer.n_ancilla == 5
        assert layer.output_dim() == 11  # 16 - 5
    
    def test_layer_forward_shape(self):
        """Forward pass returns correct shape."""
        layer = ql.VVRQLayer(n_qubits=3)
        z_input = torch.tensor([0.5, -0.3])
        output = layer(z_input)
        assert output.shape == (8,)
    
    def test_layer_forward_with_truncation(self):
        """Forward pass with truncation returns correct shape."""
        layer = ql.VVRQLayer(n_qubits=3, n_ancilla=3)
        z_input = torch.tensor([0.5, -0.3])
        output = layer(z_input)
        assert output.shape == (5,)
    
    def test_layer_forward_batched(self):
        """Forward pass handles batched input."""
        layer = ql.VVRQLayer(n_qubits=3)
        z_input = torch.rand(4, 2) * 2 - 1  # batch of 4, values in [-1, 1]
        output = layer(z_input)
        assert output.shape == (4, 8)
    
    def test_layer_output_is_probability(self):
        """Layer output is valid probability distribution."""
        layer = ql.VVRQLayer(n_qubits=3)
        z_input = torch.tensor([0.5, -0.3])
        output = layer(z_input)
        
        assert (output >= 0).all()
        assert torch.isclose(output.sum(), torch.tensor(1.0), atol=1e-5)
    
    def test_layer_has_trainable_weights(self):
        """Layer has trainable parameters."""
        layer = ql.VVRQLayer(n_qubits=3, n_layers=1)
        
        # Check weights exist
        assert hasattr(layer, 'weights')
        assert isinstance(layer.weights, torch.nn.Parameter)
        assert layer.weights.shape == (5,)  # 3 + 2 = 5
        assert layer.weights.requires_grad
    
    def test_layer_gradient_flow(self):
        """Gradients flow through the layer."""
        layer = ql.VVRQLayer(n_qubits=3, n_layers=1)
        z_input = torch.tensor([0.5, -0.3])
        
        # Forward pass
        output = layer(z_input)
        
        # Backward pass
        loss = output.sum()
        loss.backward()
        
        # Check gradients exist
        assert layer.weights.grad is not None
        assert not torch.all(layer.weights.grad == 0), "Gradients should be non-zero"
    
    def test_layer_deterministic(self):
        """Same input produces same output."""
        layer = ql.VVRQLayer(n_qubits=3)
        z_input = torch.tensor([0.5, -0.3])
        
        output1 = layer(z_input)
        output2 = layer(z_input)
        
        torch.testing.assert_close(output1, output2)
    
    def test_layer_repr(self):
        """Layer has informative string representation."""
        layer = ql.VVRQLayer(n_qubits=4, n_layers=2, n_ancilla=3)
        repr_str = repr(layer)
        assert 'n_qubits=4' in repr_str
        assert 'n_layers=2' in repr_str
        assert 'n_ancilla=3' in repr_str


class TestIntegration:
    """Integration tests for quantum layers with PyTorch."""
    
    @pytest.fixture(autouse=True)
    def check_pennylane(self):
        """Skip tests if PennyLane is not installed."""
        if not ql.HAS_PENNYLANE:
            pytest.skip("PennyLane not installed")
    
    def test_layer_in_sequential(self):
        """VVRQLayer can be used in nn.Sequential (with wrapper)."""
        # Note: VVRQLayer expects (2,) input, so we need a linear layer first
        class QuantumModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.pre = torch.nn.Linear(10, 2)
                self.quantum = ql.VVRQLayer(n_qubits=3)
                self.post = torch.nn.Linear(8, 5)
            
            def forward(self, x):
                z = torch.tanh(self.pre(x))  # Map to [-1, 1]
                probs = self.quantum(z)
                return self.post(probs)
        
        model = QuantumModel()
        x = torch.randn(1, 10)
        output = model(x)
        assert output.shape == (1, 5)
    
    def test_layer_optimizer_step(self):
        """Optimizer can update layer weights."""
        layer = ql.VVRQLayer(n_qubits=3)
        optimizer = torch.optim.Adam(layer.parameters(), lr=0.1)
        
        initial_weights = layer.weights.clone().detach()
        
        # Training step
        z_input = torch.tensor([0.5, -0.3])
        output = layer(z_input)
        loss = (output - torch.ones(8) / 8).pow(2).sum()  # Target: uniform distribution
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Weights should have changed
        assert not torch.allclose(layer.weights, initial_weights), "Weights should update after optimizer step"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
