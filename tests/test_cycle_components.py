"""
Unit tests for cycle_components module.
"""

import torch
import pytest

import sys
sys.path.insert(0, '..')

from cycle_components import ClassicalCycle, LatentMapper


class TestClassicalCycle:
    """Tests for ClassicalCycle MLP."""
    
    def test_basic_forward(self):
        """Basic forward pass works."""
        cycle = ClassicalCycle(
            input_dim=8,
            hidden_dims=[64, 32],
            output_dim=8
        )
        x = torch.randn(16, 8)
        out = cycle(x)
        
        assert out.shape == (16, 8)
    
    def test_output_shape_various(self):
        """Output shape matches output_dim."""
        # Different input/output dims
        cycle = ClassicalCycle(
            input_dim=3,
            hidden_dims=[16],
            output_dim=10
        )
        x = torch.randn(4, 3)
        out = cycle(x)
        
        assert out.shape == (4, 10)
    
    def test_single_hidden_layer(self):
        """Works with single hidden layer."""
        cycle = ClassicalCycle(
            input_dim=8,
            hidden_dims=[32],
            output_dim=8
        )
        x = torch.randn(8, 8)
        out = cycle(x)
        
        assert out.shape == (8, 8)
    
    def test_multiple_hidden_layers(self):
        """Works with multiple hidden layers."""
        cycle = ClassicalCycle(
            input_dim=8,
            hidden_dims=[128, 64, 32, 16],
            output_dim=8
        )
        x = torch.randn(8, 8)
        out = cycle(x)
        
        assert out.shape == (8, 8)
    
    def test_empty_hidden_layers(self):
        """Works with no hidden layers (direct linear)."""
        cycle = ClassicalCycle(
            input_dim=8,
            hidden_dims=[],
            output_dim=4
        )
        x = torch.randn(8, 8)
        out = cycle(x)
        
        assert out.shape == (8, 4)
    
    def test_dropout(self):
        """Dropout is applied during training."""
        cycle = ClassicalCycle(
            input_dim=8,
            hidden_dims=[64],
            output_dim=8,
            dropout=0.5
        )
        cycle.train()
        
        x = torch.randn(100, 8)
        out1 = cycle(x)
        out2 = cycle(x)
        
        # With dropout, outputs should differ in training mode
        assert not torch.allclose(out1, out2)
    
    def test_no_dropout_in_eval(self):
        """Dropout is disabled during eval."""
        cycle = ClassicalCycle(
            input_dim=8,
            hidden_dims=[64],
            output_dim=8,
            dropout=0.5
        )
        cycle.eval()
        
        x = torch.randn(16, 8)
        out1 = cycle(x)
        out2 = cycle(x)
        
        # Without dropout, outputs should be identical
        assert torch.allclose(out1, out2)
    
    def test_activation_tanh(self):
        """Tanh activation works."""
        cycle = ClassicalCycle(
            input_dim=8,
            hidden_dims=[64],
            output_dim=8,
            activation='tanh'
        )
        x = torch.randn(8, 8)
        out = cycle(x)
        
        assert out.shape == (8, 8)
    
    def test_activation_relu(self):
        """ReLU activation works."""
        cycle = ClassicalCycle(
            input_dim=8,
            hidden_dims=[64],
            output_dim=8,
            activation='relu'
        )
        x = torch.randn(8, 8)
        out = cycle(x)
        
        assert out.shape == (8, 8)
    
    def test_activation_leaky_relu(self):
        """LeakyReLU activation works."""
        cycle = ClassicalCycle(
            input_dim=8,
            hidden_dims=[64],
            output_dim=8,
            activation='leaky_relu'
        )
        x = torch.randn(8, 8)
        out = cycle(x)
        
        assert out.shape == (8, 8)
    
    def test_invalid_activation_raises(self):
        """Invalid activation raises ValueError."""
        with pytest.raises(ValueError, match="Unknown activation"):
            ClassicalCycle(
                input_dim=8,
                hidden_dims=[64],
                output_dim=8,
                activation='invalid'
            )
    
    def test_output_activation(self):
        """Output activation is applied when requested."""
        cycle = ClassicalCycle(
            input_dim=8,
            hidden_dims=[64],
            output_dim=8,
            activation='tanh',
            output_activation=True
        )
        x = torch.randn(8, 8) * 10  # Large values
        out = cycle(x)
        
        # With tanh output, values should be in [-1, 1]
        assert out.min() >= -1.0
        assert out.max() <= 1.0
    
    def test_count_parameters(self):
        """Parameter count is correct."""
        cycle = ClassicalCycle(
            input_dim=8,
            hidden_dims=[16],
            output_dim=4
        )
        # 8*16 + 16 (first linear + bias) + 16*4 + 4 (output linear + bias)
        expected = (8 * 16 + 16) + (16 * 4 + 4)
        assert cycle.count_parameters() == expected
    
    def test_gradient_flow(self):
        """Gradients flow through the network."""
        cycle = ClassicalCycle(
            input_dim=8,
            hidden_dims=[32],
            output_dim=8
        )
        x = torch.randn(4, 8, requires_grad=True)
        out = cycle(x)
        loss = out.sum()
        loss.backward()
        
        assert x.grad is not None
        assert x.grad.shape == x.shape
    
    def test_repr(self):
        """String representation is informative."""
        cycle = ClassicalCycle(
            input_dim=8,
            hidden_dims=[64, 32],
            output_dim=8
        )
        repr_str = repr(cycle)
        
        assert 'ClassicalCycle' in repr_str
        assert 'input_dim=8' in repr_str
        assert 'output_dim=8' in repr_str


class TestLatentMapper:
    """Tests for LatentMapper."""
    
    def test_linear_mapping(self):
        """Simple linear mapping works."""
        mapper = LatentMapper(input_dim=8, output_dim=128)
        x = torch.randn(16, 8)
        out = mapper(x)
        
        assert out.shape == (16, 128)
    
    def test_nonlinear_mapping(self):
        """Non-linear mapping with hidden layer works."""
        mapper = LatentMapper(
            input_dim=8,
            output_dim=128,
            hidden_dim=64
        )
        x = torch.randn(16, 8)
        out = mapper(x)
        
        assert out.shape == (16, 128)
    
    def test_dimension_reduction(self):
        """Dimension reduction works."""
        mapper = LatentMapper(input_dim=128, output_dim=8)
        x = torch.randn(16, 128)
        out = mapper(x)
        
        assert out.shape == (16, 8)
    
    def test_same_dimension(self):
        """Same input/output dimension works."""
        mapper = LatentMapper(input_dim=8, output_dim=8, hidden_dim=32)
        x = torch.randn(16, 8)
        out = mapper(x)
        
        assert out.shape == (16, 8)
    
    def test_count_parameters(self):
        """Parameter count is correct."""
        # Linear only: 8*128 + 128 = 1152
        mapper = LatentMapper(input_dim=8, output_dim=128)
        assert mapper.count_parameters() == 8 * 128 + 128
        
        # With hidden: (8*64 + 64) + (64*128 + 128) = 576 + 8320 = 8896
        mapper2 = LatentMapper(input_dim=8, output_dim=128, hidden_dim=64)
        assert mapper2.count_parameters() == (8 * 64 + 64) + (64 * 128 + 128)


class TestIntegration:
    """Integration tests combining components."""
    
    def test_quantum_to_generator_pipeline(self):
        """Simulates quantum output -> mapper -> cycle."""
        # Simulate 8-qubit quantum circuit output
        quantum_output = torch.randn(16, 8)  # batch of 16
        
        # Map to larger dimension
        mapper = LatentMapper(input_dim=8, output_dim=128, hidden_dim=64)
        mapped = mapper(quantum_output)
        
        # Process through cycle
        cycle = ClassicalCycle(
            input_dim=128,
            hidden_dims=[256, 128],
            output_dim=128,
            activation='tanh'
        )
        processed = cycle(mapped)
        
        assert processed.shape == (16, 128)
    
    def test_cycle_consistency(self):
        """Forward and inverse cycles can be chained."""
        # Forward cycle: 8 -> 128
        forward = ClassicalCycle(
            input_dim=8,
            hidden_dims=[32, 64],
            output_dim=128
        )
        
        # Inverse cycle: 128 -> 8
        inverse = ClassicalCycle(
            input_dim=128,
            hidden_dims=[64, 32],
            output_dim=8
        )
        
        x = torch.randn(16, 8)
        encoded = forward(x)
        decoded = inverse(encoded)
        
        assert encoded.shape == (16, 128)
        assert decoded.shape == (16, 8)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

