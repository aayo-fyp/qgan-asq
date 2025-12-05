#!/usr/bin/env python3
"""Run generator/discriminator/value checkpoints to produce and save molecules.

This script loads checkpoints for G, D and V (if provided), generates molecules
using the generator, sanitizes them, saves SMILES, a grid image of valid molecules,
and per-molecule similarity contour maps (Gasteiger charges).

Usage:
  python run_checkpoints.py --G path/to/5000-G.ckpt --D path/to/5000-D.ckpt --V path/to/5000-V.ckpt --n_valid 8

If D/V fail to load due to architecture mismatches the script will continue using
the Generator only (that's sufficient for sampling).
"""
import os
import argparse
import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import MolsToGridImage, SimilarityMaps
import matplotlib.pyplot as plt

from solver import Solver


def try_load(path, model, name):
    if not path:
        return False, f"{name} path not provided"
    if not os.path.exists(path):
        return False, f"{name} checkpoint not found: {path}"
    try:
        model.load_state_dict(torch.load(path, map_location=lambda s, l: s))
        return True, f"Loaded {name} from {path}"
    except Exception as e:
        return False, f"Failed to load {name}: {e}"


def sample_valid(solver, n_valid=8, gen_batch=16, max_attempts=None):
    """Sample until we collect n_valid strictly sanitized molecules or reach max_attempts."""
    max_attempts = max_attempts if max_attempts is not None else n_valid * 10
    valid = []
    smiles = []
    attempts = 0

    pad_idx = solver.data.atom_encoder_m.get(0, 0)

    solver.G.to(solver.device)
    solver.G.eval()

    with torch.no_grad():
        while len(valid) < n_valid and attempts < max_attempts:
            bsize = min(gen_batch, n_valid - len(valid))
            z = solver.sample_z(bsize)
            z = torch.from_numpy(z).to(solver.device).float()
            edges_logits, nodes_logits = solver.G(z)

            # hard assignments for construction
            edges_hat, nodes_hat = solver.postprocess((edges_logits, nodes_logits), 'hard_gumbel')
            edges_idx = torch.max(edges_hat, -1)[1]
            nodes_idx = torch.max(nodes_hat, -1)[1]

            for i in range(bsize):
                attempts += 1
                e = edges_idx[i].cpu().numpy()
                n = nodes_idx[i].cpu().numpy()
                # normalize node array to expected length
                if n.ndim == 0:
                    n = np.array([int(n)], dtype=np.int32)
                if n.shape[0] < solver.data.vertexes:
                    n = np.concatenate([n, np.zeros(solver.data.vertexes - n.shape[0], dtype=np.int32)])
                try:
                    mol = solver.data.matrices2mol(n, e, strict=True)
                except Exception:
                    mol = None
                if mol is None:
                    continue
                # sanitize
                try:
                    Chem.SanitizeMol(mol)
                except Exception:
                    continue

                valid.append(mol)
                smiles.append(Chem.MolToSmiles(mol))
                if len(valid) >= n_valid:
                    break

    return valid, smiles, attempts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--G', type=str, required=True, help='Path to generator checkpoint')
    parser.add_argument('--D', type=str, default=None, help='Path to discriminator checkpoint')
    parser.add_argument('--V', type=str, default=None, help='Path to value checkpoint')
    parser.add_argument('--n_valid', type=int, default=8, help='Number of valid molecules to collect')
    parser.add_argument('--max_attempts', type=int, default=None, help='Max generation attempts')
    parser.add_argument('--gen_batch', type=int, default=16, help='Generator batch size per sample loop')
    parser.add_argument('--mol_data_dir', type=str, default='data/gdb9_9nodes.sparsedataset', help='Dataset file')
    parser.add_argument('--qubits', type=int, default=8)
    parser.add_argument('--layer', type=int, default=1)
    parser.add_argument('--complexity', type=str, default='mr')
    args = parser.parse_args()

    # Build a minimal config for Solver
    class Cfg: pass
    cfg = Cfg()
    cfg.quantum = False
    cfg.patches = 1
    cfg.layer = args.layer
    cfg.qubits = args.qubits
    cfg.complexity = args.complexity
    cfg.g_conv_dim = [128] if cfg.complexity == 'mr' else [16]
    cfg.g_repeat_num = 6
    cfg.d_repeat_num = 6
    cfg.d_conv_dim = [[128, 64], 128, [128, 64]]
    cfg.lambda_cls = 1
    cfg.lambda_rec = 10
    cfg.lambda_gp = 10
    cfg.post_method = 'softmax'
    cfg.batch_size = args.n_valid
    cfg.num_iters = 1
    cfg.num_iters_decay = 2500
    cfg.g_lr = 1e-4
    cfg.d_lr = 1e-4
    cfg.dropout = 0.0
    cfg.n_critic = 5
    cfg.beta1 = 0.5
    cfg.beta2 = 0.999
    cfg.resume_iters = None
    cfg.test_iters = 5000
    cfg.use_tensorboard = False

    cfg.mol_data_dir = args.mol_data_dir
    cfg.log_dir = os.path.join(os.path.dirname(args.G), 'logs')
    cfg.model_save_dir = os.path.dirname(args.G)
    cfg.sample_dir = os.path.join(cfg.model_save_dir, 'samples')
    cfg.result_dir = os.path.join(cfg.model_save_dir, 'results')
    cfg.log_step = 10
    cfg.sample_step = 1000
    cfg.model_save_step = 1000
    cfg.lr_update_step = 500

    solver = Solver(cfg)

    # Attempt to load checkpoints
    ok, msg = try_load(args.G, solver.G, 'Generator')
    print(msg)
    ok_d, msg_d = try_load(args.D, solver.D, 'Discriminator')
    print(msg_d)
    ok_v, msg_v = try_load(args.V, solver.V, 'Value')
    print(msg_v)

    # Sample valid molecules
    valid, smiles, attempts = sample_valid(solver, n_valid=args.n_valid, gen_batch=args.gen_batch, max_attempts=args.max_attempts)
    print(f"Collected {len(valid)}/{args.n_valid} valid molecules after {attempts} attempts")

    out_dir = cfg.model_save_dir
    os.makedirs(out_dir, exist_ok=True)

    # Save SMILES list
    smiles_path = os.path.join(out_dir, 'generated_smiles.txt')
    with open(smiles_path, 'w') as f:
        for s in smiles:
            f.write(s + '\n')
    print(f"Saved SMILES -> {smiles_path}")

    # Save grid image of valid molecules
    if len(valid) > 0:
        grid = MolsToGridImage(valid, molsPerRow=min(4, len(valid)), subImgSize=(300, 300))
        grid_path = os.path.join(out_dir, 'generated_grid.png')
        grid.save(grid_path)
        print(f"Saved molecule grid -> {grid_path}")

    # Per-molecule similarity maps (Gasteiger)
    for i, mol in enumerate(valid):
        try:
            AllChem.ComputeGasteigerCharges(mol)
            contribs = [mol.GetAtomWithIdx(j).GetDoubleProp('_GasteigerCharge') for j in range(mol.GetNumAtoms())]
            fig = SimilarityMaps.GetSimilarityMapFromWeights(mol, contribs, colorMap=None, contourLines=10)
            map_path = os.path.join(out_dir, f'sim_map_{i}.png')
            fig.savefig(map_path, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved similarity map -> {map_path}")
        except Exception as e:
            print(f"Failed similarity map for molecule {i}: {e}")


if __name__ == '__main__':
    main()
