#!/usr/bin/env python3
"""
Build a repository-compatible `.sparsedataset` using the repo's
`SparseMolecularDataset.generate()` and `save()` utilities.

Usage:
  python scripts/build_sparsedataset.py --sdf path/to/dataset.sdf --out data/mydataset.sparsedataset

This version suppresses all RDKit and Python warnings so only tqdm is visible.
"""
import argparse
import os
import sys
import warnings

# ---------------------------
# Suppress RDKit + general warnings
# ---------------------------
warnings.filterwarnings("ignore")

try:
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
except Exception:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sdf', required=True, help='Input SDF file')
    parser.add_argument('--out', required=True, help='Output sparsedataset path (e.g. data/qm9.sparsedataset)')
    parser.add_argument('--add-h', action='store_true', help='Add explicit hydrogens when parsing')
    parser.add_argument('--size', type=int, default=None, help='Max number of molecules to include')
    args = parser.parse_args()

    from data.sparse_molecular_dataset import SparseMolecularDataset

    ds = SparseMolecularDataset()
    print('Generating sparse dataset from', args.sdf)

    # GENERATE WITH PROGRESS BAR ONLY
    ds.generate(args.sdf, add_h=args.add_h, size=args.size)

    out_dir = os.path.dirname(args.out)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    ds.save(args.out)
    print('Saved sparsedataset to', args.out)

if __name__ == '__main__':
    main()