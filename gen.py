#!/usr/bin/env python3
"""
Generate EXACT n_samples molecules from a saved Generator checkpoint.
- Auto-detects gen-weights row in CSV in same model dir (first column = iteration index).
- Uses PennyLane gen_circuit sampling (mirrors training's gen_circuit randomness).
- Does NOT loop until n_valid; produces exactly n_samples attempts and records valid molecules.
- Saves SMILES, a grid image of the VALID molecules, and similarity maps for each valid molecule.
"""

import os
import argparse
import csv
import math
import time
import numpy as np
import torch
import random
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import MolsToGridImage, SimilarityMaps
import matplotlib.pyplot as plt

# PennyLane
import pennylane as qml

# project imports (assumes you're running from repo root)
from solver import Solver
from data.sparse_molecular_dataset import SparseMolecularDataset

# ------------ Helpers -------------
def find_weights_for_index(model_dir, index):
    """Scan csv files in model_dir for a row whose first column equals index.
       Returns list of float weights if found, else None.
    """
    import re

    for fn in os.listdir(model_dir):
        if not fn.lower().endswith('.csv'):
            continue
        path = os.path.join(model_dir, fn)

        try:
            with open(path, 'r') as fh:
                reader = csv.reader(fh)
                for row in reader:
                    if not row:
                        continue

                    cell0 = row[0].strip()

                    m = re.match(r'^(\d+)', cell0)
                    if not m:
                        continue
                    first = int(m.group(1))
                    if first != index:
                        continue

                    floats = []

                    if len(row) > 1:
                        for v in row[1:]:
                            try:
                                floats.append(float(v))
                            except:
                                pass
                    else:
                        rest = cell0[len(m.group(1)):]
                        candidates = [x for x in rest.replace(',', ' ').split() if x]
                        for v in candidates:
                            try:
                                floats.append(float(v))
                            except:
                                pass

                    if len(floats) > 0:
                        return floats
        except Exception:
            continue

    return None

def parse_index_from_ckpt_path(ckpt):
    """Common formats: '.../13480-G.ckpt' or '13480-G.ckpt'"""
    base = os.path.basename(ckpt)
    # find leading integer substring
    prefix = base.split('-')[0]
    try:
        return int(prefix)
    except Exception:
        # fallback: numbers in filename
        import re
        m = re.search(r'(\d+)', base)
        if m:
            return int(m.group(1))
    return None

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

# ------------ Main -------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--G', required=True, help='Path to generator checkpoint (e.g. models/13480-G.ckpt)')
    parser.add_argument('--n_samples', type=int, default=50, help='Number of samples to generate (exactly)')
    parser.add_argument('--out_dir', type=str, default=None, help='Output folder (default: <model_dir>/generated)')
    parser.add_argument('--mol_data_dir', type=str, default='data/gdb9_9nodes.sparsedataset', help='dataset file used to init encoders')
    parser.add_argument('--qubits', type=int, default=8, help='number of qubits used in gen_circuit')
    parser.add_argument('--layer', type=int, default=1, help='quantum layer count (training value)')
    parser.add_argument('--complexity', type=str, default='nr', choices=['mr','hr','nr'], help='model complexity used at training')
    parser.add_argument('--post_method', type=str, default='hard_gumbel', choices=['soft_gumbel','softmax','hard_gumbel'])
    parser.add_argument('--seed', type=int, default=None, help='optional random seed for reproducible sampling (affects gen_circuit random uniforms)')
    args = parser.parse_args()

    G_path = args.G
    if not os.path.exists(G_path):
        raise SystemExit(f"Generator checkpoint not found: {G_path}")

    model_dir = os.path.dirname(G_path)
    out_dir = args.out_dir or os.path.join(model_dir, 'generated')
    ensure_dir(out_dir)

    # Optional RNG seed
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    # Build minimal config required by Solver (mirrors training config choices)
    class Cfg: pass
    cfg = Cfg()
    cfg.quantum = False   # IMPORTANT: For sampling we feed precomputed z vectors directly to G (like your sampling scripts)
    cfg.patches = 1
    cfg.layer = args.layer
    cfg.qubits = args.qubits
    cfg.complexity = args.complexity
    # g_conv_dim must match complexity used during training when model was saved
    if cfg.complexity == 'mr':
        cfg.g_conv_dim = [128]
    elif cfg.complexity == 'hr':
        cfg.g_conv_dim = [16]
    elif cfg.complexity == 'nr':
        cfg.g_conv_dim = [128, 256, 512]
    cfg.g_repeat_num = 6
    cfg.d_repeat_num = 6
    cfg.d_conv_dim = [[128, 64], 128, [128, 64]]
    cfg.lambda_cls = 1
    cfg.lambda_rec = 10
    cfg.lambda_gp = 10
    cfg.post_method = args.post_method
    cfg.batch_size = args.n_samples
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
    cfg.log_dir = os.path.join(model_dir, 'logs')
    cfg.model_save_dir = model_dir
    cfg.sample_dir = os.path.join(model_dir, 'samples')
    cfg.result_dir = os.path.join(model_dir, 'results')
    cfg.log_step = 10
    cfg.sample_step = 1000
    cfg.model_save_step = 1000
    cfg.lr_update_step = 500
    cfg.z_dim = cfg.qubits  # generator input dimension (z_dim)

    print("Building Solver and loading dataset (this uses the dataset encoders to rebuild molecules)...")
    solver = Solver(cfg)  # Solver will initialize Generator with same shapes as during training

    # Load generator checkpoint (lenient load -> strict=False)
    print(f"Loading generator checkpoint: {G_path}")
    ckpt = torch.load(G_path, map_location='cpu')
    # Some checkpoints store state_dict, some are raw dict: handle both cases
    if isinstance(ckpt, dict) and all(k.startswith('layers') or k in ('edges_layer.weight','nodes_layer.weight', 'edges_layer.bias','nodes_layer.bias') for k in ckpt.keys()):
        state_dict = ckpt
    else:
        # if checkpoint is wrapped (like {'state_dict': ...}) try common keys
        if isinstance(ckpt, dict) and 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt

    # Apply to generator
    try:
        solver.G.load_state_dict(state_dict, strict=False)
        print("Loaded generator state_dict (strict=False).")
    except Exception as e:
        # still try partial load (already strict=False) but report error
        print("Warning: generator load_state_dict raised exception:", e)
        solver.G.load_state_dict(state_dict, strict=False)
        print("Attempted lenient load.")

    # ---------- find gen_weights automatically using index ----------
    idx = parse_index_from_ckpt_path(G_path)
    gen_weights = None
    if idx is not None:
        found = find_weights_for_index(model_dir, idx)
        if found is not None:
            gen_weights = np.array(found, dtype=np.float32)
            print(f"Found gen_weights for index {idx} in CSV (len={len(gen_weights)})")
        else:
            print(f"No CSV row found for index {idx} in {model_dir}; will sample random gen_weights.")
    else:
        print("Could not parse index from checkpoint filename; will sample random gen_weights.")

    # If not found, sample random weights of expected length:
    expected_len = cfg.layer * (cfg.qubits*2 - 1)  # matches training formula used in your training file
    if gen_weights is None:
        gen_weights = (np.random.rand(expected_len) * 2 * np.pi - np.pi).astype(np.float32)
        print(f"Sampled random gen_weights of length {expected_len}")

    # --- handle mismatch between CSV length and expected length ---
    if gen_weights is not None and len(gen_weights) != expected_len:
        print(f"[WARN] gen_weights length {len(gen_weights)} != expected {expected_len}. Fixing length...")
        if len(gen_weights) < expected_len:
            # pad with random angles
            pad_count = expected_len - len(gen_weights)
            pad_vals = (np.random.rand(pad_count) * 2 * np.pi - np.pi).astype(np.float32)
            gen_weights = np.concatenate([gen_weights, pad_vals])
            print(f" -> padded gen_weights to length {expected_len}")
        else:
            # truncate
            gen_weights = gen_weights[:expected_len]
            print(f" -> truncated gen_weights to length {expected_len}")

    # Create PennyLane device + qnode identical to training gen_circuit
    dev = qml.device('default.qubit', wires=cfg.qubits)

    @qml.qnode(dev, interface='torch')
    def gen_circuit(w):
        # training used random.uniform(-1,1) per call to gen_circuit -> keep the same
        z1 = random.uniform(-1, 1)
        z2 = random.uniform(-1, 1)

        for i in range(cfg.qubits):
            qml.RY(np.arcsin(z1), wires=i)
            qml.RZ(np.arcsin(z2), wires=i)

        # apply variational layers using elements of w
        # w length expected = cfg.layer*(cfg.qubits*2 - 1)
        # training used: for each layer, RY(w[i]) across qubits then for i in range(qubits-1):
        #   CNOT; RZ(w[i+qubits]); CNOT
        start = 0
        for l in range(cfg.layer):
            # single-qubit RY on each wire with first qubits parameters
            for i in range(cfg.qubits):
                qml.RY(w[start + i], wires=i)
            # entangling style with RZ on target using next qubits parameters
            for i in range(cfg.qubits - 1):
                qml.CNOT(wires=[i, i+1])
                qml.RZ(w[start + cfg.qubits + i], wires=i+1)
                qml.CNOT(wires=[i, i+1])
            start += (cfg.qubits*2 - 1)

        return [qml.expval(qml.PauliZ(i)) for i in range(cfg.qubits)]

    # helper to sample z vectors from gen_circuit
    def sample_z_from_gen_circuit(gen_circuit, gw, n):
        samples = []
        # gw must be vector-like; convert to torch tensor accepted by qnode
        import torch as _torch
        w_t = _torch.tensor(gw, dtype=_torch.float32)
        for _ in range(n):
            try:
                out = gen_circuit(w_t)
            except Exception as e:
                # qnode might return a list of numpy floats or torch tensor
                raise RuntimeError(f"gen_circuit failed: {e}")
            # normalize output types to numpy 1D float array
            if isinstance(out, _torch.Tensor):
                arr = out.detach().cpu().numpy().astype(np.float32)
            elif isinstance(out, (list, tuple, np.ndarray)):
                arr = np.array(out, dtype=np.float32)
            else:
                arr = np.array(out, dtype=np.float32)
            samples.append(arr)
        if len(samples) == 0:
            raise RuntimeError("No samples collected from gen_circuit.")
        return np.stack(samples, axis=0).astype(np.float32)

    print(f"Sampling {args.n_samples} quantum z vectors from gen_circuit (qubits={cfg.qubits}, layer={cfg.layer}) ...")
    z_np = sample_z_from_gen_circuit(gen_circuit, gen_weights, args.n_samples)
    # convert to torch
    z_t = torch.from_numpy(z_np).to(torch.float32)

    # Run generator forward pass in batches (to match solver.G expecting batch dimension)
    solver.G.to('cpu')
    solver.G.eval()

    valid_mols = []
    smiles_list = []
    attempts = args.n_samples

    with torch.no_grad():
        batch = z_t
        if batch.dim() == 1:
            batch = batch.unsqueeze(0)
        # forward
        edges_logits, nodes_logits = solver.G(batch)
        # apply chosen postprocess (use solver.postprocess)
        edges_post, nodes_post = solver.postprocess((edges_logits, nodes_logits), args.post_method)

        # If postprocess returns logits / soft one-hot, convert to hard indices
        try:
            edges_idx = torch.max(edges_post, -1)[1]
        except Exception:
            edges_idx = edges_post
        try:
            nodes_idx = torch.max(nodes_post, -1)[1]
        except Exception:
            nodes_idx = nodes_post

        # For each sample, convert & sanitize
        for i in range(edges_idx.shape[0]):
            e = edges_idx[i].cpu().numpy()
            n = nodes_idx[i].cpu().numpy()
            # normalize node length
            if n.ndim == 0:
                n = np.array([int(n)], dtype=np.int32)
            if n.shape[0] < solver.data.vertexes:
                n = np.concatenate([n, np.zeros(solver.data.vertexes - n.shape[0], dtype=np.int32)])
            mol = None
            try:
                mol = solver.data.matrices2mol(n, e, strict=True)
            except Exception:
                mol = None
            if mol is None:
                smiles_list.append(None)
                valid_mols.append(None)
            else:
                # sanitize
                try:
                    Chem.SanitizeMol(mol)
                except Exception:
                    smiles_list.append(None)
                    valid_mols.append(None)
                    continue
                s = Chem.MolToSmiles(mol)
                smiles_list.append(s)
                valid_mols.append(mol)
                print(f"Collected valid molecule #{len([m for m in valid_mols if m is not None])} -> {s}")

    # Save SMILES for all samples (None = invalid placeholder)
    smi_path = os.path.join(out_dir, 'generated_smiles_all.txt')
    with open(smi_path, 'w') as fh:
        for s in smiles_list:
            fh.write((s if s is not None else 'INVALID') + '\n')
    print(f"Saved SMILES (including INVALID placeholders) -> {smi_path}")


    # Save validity scores (1 = valid molecule, 0 = invalid)
    validity_path = os.path.join(out_dir, 'validity_scores.txt')
    with open(validity_path, 'w') as vf:
        for m in valid_mols:
            vf.write(('1' if m is not None else '0') + '\n')
    print(f"Saved validity scores -> {validity_path}")

    # Save RDKit metrics for each molecule
    metrics_path = os.path.join(out_dir, 'metrics.txt')
    from rdkit.Chem import Descriptors
    from rdkit.Chem import QED
    try:
        from rdkit.Chem import rdMolDescriptors
        sascore_fn = rdMolDescriptors.CalcSyntheticAccessibilityScore
    except:
        sascore_fn = None

    with open(metrics_path, 'w') as mf:
        mf.write("Index\tValid\tSMILES\tQED\tSAS\tMolWt\tLogP\tNumRings\n")
        for idx, mol in enumerate(valid_mols):
            if mol is None:
                mf.write(f"{idx}\t0\tINVALID\t0\t0\t0\t0\t0\n")
                continue

            smi = smiles_list[idx]
            qed = QED.qed(mol)
            try:
                sas = sascore_fn(mol) if sascore_fn is not None else 0
            except:
                sas = 0
            molwt = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            rings = rdMolDescriptors.CalcNumRings(mol)

            mf.write(f"{idx}\t1\t{smi}\t{qed:.4f}\t{sas:.4f}\t{molwt:.2f}\t{logp:.3f}\t{rings}\n")

    print(f"Saved RDKit metric scores -> {metrics_path}")

    # Save grid of valid molecules (if any)
    valid_nonnull = [m for m in valid_mols if m is not None]
    if len(valid_nonnull) > 0:
        grid = MolsToGridImage(valid_nonnull, molsPerRow=min(4, len(valid_nonnull)), subImgSize=(300,300))
        grid_path = os.path.join(out_dir, 'generated_grid.png')
        grid.save(grid_path)
        print(f"Saved molecule grid -> {grid_path}")
    else:
        print("No valid molecules to create grid image.")

    # Similarity maps for each valid molecule
    for i, mol in enumerate(valid_mols):
        if mol is None:
            print(f"Sample {i}: invalid molecule — skipping similarity map")
            continue
        try:
            AllChem.ComputeGasteigerCharges(mol)
            contribs = []
            for j in range(mol.GetNumAtoms()):
                try:
                    c = mol.GetAtomWithIdx(j).GetDoubleProp('_GasteigerCharge')
                except Exception:
                    c = 0.0
                contribs.append(c)
            fig = SimilarityMaps.GetSimilarityMapFromWeights(mol, contribs, colorMap=None, contourLines=10)
            map_path = os.path.join(out_dir, f'sim_map_{i}.png')
            fig.savefig(map_path, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved similarity map -> {map_path}")
        except Exception as e:
            print(f"Failed similarity map for sample {i}: {e}")

    print(f"Done. Produced {args.n_samples} samples (valid={len(valid_nonnull)}). Outputs in: {out_dir}")

if __name__ == '__main__':
    main()