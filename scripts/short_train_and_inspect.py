import time
import os
import numpy as np
import torch
import torch.nn.functional as F
from types import SimpleNamespace
from solver import Solver
from data.sparse_molecular_dataset import SparseMolecularDataset
from utils import MolecularMetrics
from rdkit import Chem

# Minimal config matching expected Solver attributes
config = SimpleNamespace()
config.qubits = 8
config.g_conv_dim = [128]
config.d_conv_dim = [[128, 64], 128, [128, 64]]
config.g_repeat_num = 6
config.d_repeat_num = 6
config.lambda_cls = 1
config.lambda_rec = 10
config.lambda_gp = 10
config.post_method = 'softmax'
config.batch_size = 2
config.num_iters = 2
config.num_iters_decay = 2500
config.g_lr = 0.0001
config.d_lr = 0.0001
config.dropout = 0.0
config.n_critic = 1
config.beta1 = 0.5
config.beta2 = 0.999
config.resume_iters = None
config.test_iters = 5000
config.use_tensorboard = False
config.log_dir = 'tmp_short/logs'
config.model_save_dir = 'tmp_short/models'
config.sample_dir = 'tmp_short/samples'
config.result_dir = 'tmp_short/results'
config.log_step = 1
config.sample_step = 1000
config.model_save_step = 1000
config.lr_update_step = 500
config.metric = 'validity,sas'
# cycle/qdi args
config.cycle = 'classical'
config.lambda_cycle = 0.0
config.qdi_reps = 2
config.qdi_layers = 1
config.qdi_batch = False
config.patches = 1
config.layer = 1
config.quantum = False
config.complexity = 'mr'
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
config.mol_data_dir = os.path.join(base_dir, 'data', 'gdb9.sparsedataset')
config.use_tensorboard = False

print('Building solver and running short training...')
solver = Solver(config)
# Run training (this will print log_step outputs)
solver.train()

# After training, inspect rewards for one generated batch
print('\nInspecting rewards after short training:')
# get a train batch
mols_real, _, _, a, x, _, _, _, _ = solver.data.next_train_batch(solver.batch_size)
z = solver.sample_z(solver.batch_size)
z_t = torch.from_numpy(z).float().to(solver.device)
with torch.no_grad():
    edges_logits, nodes_logits = solver.G(z_t)
    (edges_hard, nodes_hard) = solver.postprocess((edges_logits, nodes_logits), 'hard_gumbel')
    edges_idx = torch.max(edges_hard, -1)[1]
    nodes_idx = torch.max(nodes_hard, -1)[1]
    mols_fake = [solver.data.matrices2mol(n_.cpu().numpy(), e_.cpu().numpy(), strict=True) for e_, n_ in zip(edges_idx, nodes_idx)]

rewardR = solver.reward(mols_real).reshape(-1)
rewardF = solver.reward(mols_fake).reshape(-1)
sa_real = MolecularMetrics.synthetic_accessibility_score_scores(mols_real, norm=True)
sa_fake = MolecularMetrics.synthetic_accessibility_score_scores(mols_fake, norm=True)
valid_real = MolecularMetrics.valid_scores(mols_real)
valid_fake = MolecularMetrics.valid_scores(mols_fake)

print('Real (valid, SA, reward):')
for i in range(len(mols_real)):
    s = Chem.MolToSmiles(mols_real[i]) if mols_real[i] is not None else None
    print(f'  [{i}] smiles={s}, valid={valid_real[i]:.3f}, SA={sa_real[i]:.3f}, rewardR={rewardR[i]:.3f}')

print('\nFake (valid, SA, reward):')
for i in range(len(mols_fake)):
    s = Chem.MolToSmiles(mols_fake[i]) if mols_fake[i] is not None else None
    print(f'  [{i}] smiles={s}, valid={valid_fake[i]:.3f}, SA={sa_fake[i]:.3f}, rewardF={rewardF[i]:.3f}')

print('\nSummary arrays:')
print('rewardR =', rewardR)
print('rewardF =', rewardF)
