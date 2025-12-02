"""
Unit tests for quantum_layers module.
"""

import numpy as np
import pytest

import sys
sys.path.insert(0, '..')

import quantum_layers as ql


class TestVQCHelpers:
    """Tests for helper functions."""
    
    def test_vqc_output_dim_default(self):
        """Output dimension equals 2**n_qubits."""
        assert ql.vqc_output_dim() == 2 ** 3
        assert ql.vqc_output_dim(n_qubits=3) == 2 ** 3
    
    def test_vqc_output_dim_various(self):
        """Output dimension scales with qubits."""
        assert ql.vqc_output_dim(n_qubits=1) == 2
        assert ql.vqc_output_dim(n_qubits=5) == 2 ** 5
        assert ql.vqc_output_dim(n_qubits=8) == 2 ** 8
    
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


class TestVVRQCircuit:
    """Tests for the VVRQ quantum circuit."""
    
    @pytest.fixture(autouse=True)
    def check_pennylane(self):
        """Skip tests if PennyLane is not installed."""
        if not ql.HAS_PENNYLANE:
            pytest.skip("PennyLane not installed")
    
    def test_vvrq_output_shape_default(self):
        """Default circuit outputs 2**3 probability values."""
        weights = np.random.rand(5) * 0.1
        output = ql.vvrq_vqc(weights)
        assert output.shape == (2 ** 3,)
    
    def test_vvrq_output_shape_various(self):
        """Output shape matches n_qubits."""
        # 4 qubits
        weights = np.random.rand(7) * 0.1
        output = ql.vvrq_vqc(weights, n_qubits=4)
        assert output.shape == (2 ** 4,)
        
        # 1 qubit
        weights = np.random.rand(1) * 0.1
        output = ql.vvrq_vqc(weights, n_qubits=1)
        assert output.shape == (2,)
    
    def test_vvrq_output_range(self):
        """Probabilities are in [0, 1] and sum to 1."""
        weights = np.random.rand(5) * 0.1
        output = ql.vvrq_vqc(weights, n_qubits=3)

        assert (output >= 0.0).all(), f"Values below 0: {output}"
        assert (output <= 1.0).all(), f"Values above 1: {output}"
        assert abs(output.sum() - 1.0) < 1e-6
    
    def test_vvrq_deterministic_with_z_input(self):
        """Same z_input produces same output for same weights."""
        weights = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        z_input = (0.5, -0.3)

        output1 = ql.vvrq_vqc(weights, z_input=z_input)
        output2 = ql.vvrq_vqc(weights, z_input=z_input)

        np.testing.assert_array_almost_equal(output1, output2)
    
    def test_vvrq_different_weights_different_output(self):
        """Different weights produce different outputs."""
        z_input = (0.5, -0.3)
        
        weights1 = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        weights2 = np.array([0.5, 0.4, 0.3, 0.2, 0.1])
        
        output1 = ql.vvrq_vqc(weights1, z_input=z_input)
        output2 = ql.vvrq_vqc(weights2, z_input=z_input)

        assert not np.allclose(output1, output2), "Different weights should produce different outputs"
    
    def test_vvrq_insufficient_weights_raises(self):
        """Raises ValueError if not enough weights provided."""
        weights = np.array([0.1, 0.2])  # Only 2, need 5 for 3 qubits

        with pytest.raises(ValueError, match="Expected at least 5 weights"):
            ql.vvrq_vqc(weights, n_qubits=3)
    
    def test_vvrq_extra_weights_ignored(self):
        """Extra weights beyond required are ignored."""
        weights = np.random.rand(10) * 0.1  # 10 weights, only 5 needed
        z_input = (0.5, -0.3)

        output = ql.vvrq_vqc(weights, n_qubits=3, z_input=z_input)
        assert output.shape == (2 ** 3,)
    
    def test_vvrq_multiple_layers(self):
        """Circuit works with multiple layers."""
        # 3 qubits, 2 layers = 10 weights
        weights = np.random.rand(10) * 0.1
        output = ql.vvrq_vqc(weights, n_qubits=3, n_layers=2)

        assert output.shape == (2 ** 3,)
        assert (output >= 0.0).all()
        assert (output <= 1.0).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

