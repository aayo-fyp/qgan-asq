import os,sys
proj = os.path.join(os.getcwd())
sys.path.insert(0, proj)
from models import Generator
import torch
print('models imported')
print('HAS_QUANTUM in models module:', getattr(__import__('models'), 'HAS_QUANTUM', None))
conv_dims=[16,32]; z_dim=8; vertexes=5; edges=4; nodes=6
try:
    g = Generator(conv_dims=conv_dims, z_dim=z_dim, vertexes=vertexes, edges=edges, nodes=nodes, dropout=0.0, quantum=True, vqc_kwargs={'n_qubits':3,'n_layers':1,'n_ancilla':1})
    print('g.quantum', g.quantum)
    print('has vqc attr:', hasattr(g, 'vqc'))
    if hasattr(g, 'vqc'):
        print('vqc weight shape:', g.vqc.weights.shape)
    z_q = torch.randn(2,3)
    edges_logits, nodes_logits = g(z_q)
    print('edges_logits.shape', edges_logits.shape)
    print('nodes_logits.shape', nodes_logits.shape)
    loss = edges_logits.abs().sum() + nodes_logits.abs().sum()
    loss.backward()
    print('grad on vqc weights exists?', g.vqc.weights.grad is not None)
    if g.vqc.weights.grad is not None:
        print('grad mean abs:', g.vqc.weights.grad.abs().mean().item())
except Exception as e:
    print('ERROR', e)
    raise
