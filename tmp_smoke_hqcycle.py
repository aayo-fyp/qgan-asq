import torch

from cycle_components import HQCycle

# build HQCycle with defaults
hc = HQCycle(input_dim=5*14 + 5*6, intermediate_dim=64, z_dim=8, classical_hidden=[128,64], qdi_kwargs={'n_reps':2, 'n_layers':1})
print('HQCycle created:', hc)

# create dummy flattened graph input: batch x input_dim
x = torch.randn(2, hc.input_dim)

z = hc(x)
print('z shape:', z.shape)

# simple loss and backward
loss = z.pow(2).sum()
loss.backward()

# check gradients on qdi weights
has_grad = hasattr(hc.qdi, 'weights') and hc.qdi.weights.grad is not None
print('qdi weights grad exists?', has_grad)
if has_grad:
    print('grad mean abs:', hc.qdi.weights.grad.abs().mean().item())
