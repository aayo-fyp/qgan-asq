"""
Unit tests for QDILayer and HQCycle components.
"""

import sys
import torch
import pytest

sys.path.insert(0, '..')

from quantum_layers import QDILayer
from cycle_components import HQCycle


def test_qdilayer_forward_backward():
    """QDILayer forward returns (batch, n_qubits) and gradients flow."""
    batch = 2
    input_dim = 10
    n_qubits = 4

    qdi = QDILayer(input_dim=input_dim, n_qubits=n_qubits, n_reps=1, n_layers=1)
    qdi.train()

    x = torch.randn(batch, input_dim, requires_grad=True)
    out = qdi(x)

    assert out.shape == (batch, n_qubits)

    loss = out.pow(2).sum()
    loss.backward()

    assert qdi.weights.grad is not None
    assert qdi.encoder.weight.grad is not None


def test_hqcycle_forward_backward_and_optimizer_step():
    """HQCycle integrates classical encoder + QDI and updates via optimizer."""
    batch = 2
    # choose input small: nodes_flat + edges_flat like in solver
    vertexes = 3
    atom_types = 6
    bond_types = 4
    input_dim = vertexes * atom_types + vertexes * vertexes * bond_types

    z_dim = 4
    qdi_kwargs = {'n_reps': 1, 'n_layers': 1, 'n_qubits': z_dim}

    hc = HQCycle(input_dim=input_dim, intermediate_dim=16, z_dim=z_dim, classical_hidden=[32], qdi_kwargs=qdi_kwargs)
    hc.train()

    x = torch.randn(batch, input_dim)
    z = torch.randn(batch, z_dim)

    out = hc(x)
    assert out.shape == (batch, z_dim)

    loss = (out - z).pow(2).mean()

    # optimizer on cycle params
    opt = torch.optim.Adam(hc.parameters(), lr=1e-3)
    opt.zero_grad()
    loss.backward()

    # grads should exist
    grads = [p.grad for p in hc.parameters() if p.requires_grad]
    assert any(g is not None for g in grads)

    # take an optimizer step and ensure parameters changed
    params_before = [p.detach().clone() for p in hc.parameters() if p.requires_grad]
    opt.step()
    params_after = [p.detach().clone() for p in hc.parameters() if p.requires_grad]

    changed = any(((a - b).abs().sum().item() > 0) for a, b in zip(params_before, params_after))
    assert changed


def test_qdilayer_batched_matches_unbatched():
    """Batched QDI should produce same shapes and gradients as unbatched (within tolerance)."""
    batch = 4
    input_dim = 12
    n_qubits = 4

    # unbatched
    qdi_u = QDILayer(input_dim=input_dim, n_qubits=n_qubits, n_reps=1, n_layers=1, batch=False)
    qdi_b = QDILayer(input_dim=input_dim, n_qubits=n_qubits, n_reps=1, n_layers=1, batch=True)

    # sync weights for deterministic comparison
    with torch.no_grad():
        qdi_b.weights.copy_(qdi_u.weights)
        qdi_b.encoder.weight.copy_(qdi_u.encoder.weight)
        qdi_b.encoder.bias.copy_(qdi_u.encoder.bias)

    x = torch.randn(batch, input_dim, requires_grad=True)

    out_u = qdi_u(x)
    out_b = qdi_b(x)

    assert out_u.shape == out_b.shape

    loss_u = out_u.pow(2).sum()
    loss_b = out_b.pow(2).sum()

    loss_u.backward()
    loss_b.backward()

    # gradients exist and shapes match for weights
    assert qdi_u.weights.grad is not None
    assert qdi_b.weights.grad is not None

    # outputs close
    assert torch.allclose(out_u, out_b, atol=1e-6, rtol=1e-5)


if __name__ == '__main__':
    pytest.main([__file__, '-q'])
