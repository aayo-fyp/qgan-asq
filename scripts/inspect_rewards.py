import numpy as np
import torch
import torch.nn.functional as F
from data.sparse_molecular_dataset import SparseMolecularDataset
from utils import MolecularMetrics
from models import Generator
from rdkit import Chem

# Config-like defaults matching solver defaults
BATCH = 2
Z_DIM = 8
G_CONV_DIM = [128]
DROPOUT = 0.0

# Load data
data = SparseMolecularDataset()
data.load('data/gdb9.sparsedataset')

# Get a small train batch
mols, _, _, a, x, _, _, _, _ = data.next_train_batch(BATCH)
print(f"Loaded {len(mols)} real molecules (batch={BATCH})")

# Compute per-sample metrics for real molecules
sa_real = MolecularMetrics.synthetic_accessibility_score_scores(mols, norm=True)
valid_real = MolecularMetrics.valid_scores(mols)
reward_real = valid_real * sa_real

print("Real molecules:")
for i, m in enumerate(mols):
    s = Chem.MolToSmiles(m) if m is not None else None
    print(f"  [{i}] smiles={s}, valid={valid_real[i]:.3f}, SA(norm)={sa_real[i]:.3f}, reward={reward_real[i]:.3f}")

# Build an untrained generator and sample fake molecules
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
G = Generator(G_CONV_DIM, Z_DIM, data.vertexes, data.bond_num_types, data.atom_num_types, DROPOUT).to(device)

z = np.random.normal(0, 1, size=(BATCH, Z_DIM))
z_t = torch.from_numpy(z).float().to(device)

with torch.no_grad():
    edges_logits, nodes_logits = G(z_t)
    # Hard gumbel to discrete matrices
    edges_hard = F.gumbel_softmax(edges_logits.contiguous().view(-1, edges_logits.size(-1)), hard=True)
    nodes_hard = F.gumbel_softmax(nodes_logits.contiguous().view(-1, nodes_logits.size(-1)), hard=True)
    edges_hard = edges_hard.view(edges_logits.size())
    nodes_hard = nodes_hard.view(nodes_logits.size())
    edges_idx = torch.max(edges_hard, -1)[1]
    nodes_idx = torch.max(nodes_hard, -1)[1]

    # Convert to RDKit molecules
    mols_fake = [data.matrices2mol(n_.cpu().numpy(), e_.cpu().numpy(), strict=True)
                 for e_, n_ in zip(edges_idx, nodes_idx)]

# Compute metrics for fake molecules
sa_fake = MolecularMetrics.synthetic_accessibility_score_scores(mols_fake, norm=True)
valid_fake = MolecularMetrics.valid_scores(mols_fake)
reward_fake = valid_fake * sa_fake

print("\nFake molecules (untrained generator):")
for i, m in enumerate(mols_fake):
    s = Chem.MolToSmiles(m) if m is not None else None
    print(f"  [{i}] smiles={s}, valid={valid_fake[i]:.3f}, SA(norm)={sa_fake[i]:.3f}, reward={reward_fake[i]:.3f}")

# Print arrays
print('\nSummary arrays:')
print('valid_real =', valid_real)
print('sa_real    =', sa_real)
print('rewardR    =', reward_real.reshape(-1))
print('valid_fake =', valid_fake)
print('sa_fake    =', sa_fake)
print('rewardF    =', reward_fake.reshape(-1))
