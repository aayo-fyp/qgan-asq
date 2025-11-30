## Purpose
Short, focused guidance for AI coding agents working in this repo (quantum-gan). Use this to make quick, correct edits, tests and run changes.

## Big picture (what this repo does)
- Implements MolGAN-style generative models for small molecules with an optional hybrid quantum generator (PennyLane + PyTorch).
- Key concepts: classical Generator/Discriminator (`models.py`), training/management logic in `solver.py`, dataset handling in `data/sparse_molecular_dataset.py`, and quantum generator circuits embedded in `main.py`, `p2_qgan_hg.py`, and `p4_qgan_hg.py`.

## Entrypoints & common runs
- Training (quantum mode): `python main.py --quantum True --layer 2 --qubits 10 --complexity hr` (see `README.md`).
- Patched quantum variants: run `p2_qgan_hg.py` or `p4_qgan_hg.py` for 2-/4-patch generators.
- Defaults write artifacts under `qgan-hg-{complexity}-q{qubits}-l{layer}/` (logs, models, samples, results).
- To resume: pass `--resume_iters <N>`; checkpoints are `{N}-G.ckpt`, `{N}-D.ckpt`, `{N}-V.ckpt` in the model directory.

## Architecture & dataflow (concise)
- Data: `data/sparse_molecular_dataset.py` loads and exposes `.data`, `.smiles`, `.vertexes`, `.atom_num_types`, `.bond_num_types` used by `Solver`.
- Solver: `solver.py` builds `G`, `D`, `V` and contains training loop helpers (postprocess, reward, gradient penalty). Prefer edits here for training logic changes.
- Models: `models.py` defines `Generator` and `Discriminator` using graph conv layers from `layers.py` (look for `GraphConvolution`, `GraphAggregation`).
- Quantum integration: `main.py` (and `p2_qgan_hg.py` / `p4_qgan_hg.py`) define `qml.qnode` functions (e.g. `gen_circuit`) and pass sampled quantum circuit outputs as `z` to the `Generator` when `--quantum True`.

## Project-specific conventions & patterns
- Postprocessing: generator outputs go through `postprocess(..., method)` in `Solver` using `softmax`, `soft_gumbel`, or `hard_gumbel` — this is how logits become discrete molecule graphs.
- Reward & metrics: `MolecularMetrics` (in `utils.py`) computes validity, QED, SAS, novelty, diversity and is multiplied into the reward used to train `V`/G.
- Checkpointing: models saved with step-prefix `'{iter}-G.ckpt'` etc. Weight history CSVs (e.g. `molgan_red_weights.csv`, `gen_weights`) are appended in the model folder.
- Quantum patching: `main.py` asserts `config.patches == 1`; use `p2_qgan_hg.py` and `p4_qgan_hg.py` for patched circuit versions which concatenate qnode outputs.

## Dependencies and gotchas to mention in PRs/edits
- Native heavy deps: `rdkit` required for molecule ops; `pennylane` required for qnodes; `frechetdist` used for Fréchet distance (fd) metric. README also lists `tensorflow==1.15` (some older helper code expects TF1 APIs elsewhere).
- PennyLane qnode interface is `interface='torch'` — keep PyTorch tensors/gradients compatible when editing circuit outputs.
- The code uses `torch.device('cuda' if torch.cuda.is_available() else 'cpu')`; tests should account for both CPU and GPU runs. Unit tests should set deterministic seeds and limit GPU requirements.

## Where to look for quick edits/bugs
- Training loop & quantum injection: `main.py`, `p2_qgan_hg.py`, `p4_qgan_hg.py` — edit these to change circuit layout, number of qubits, or how quantum outputs map to `z`.
- Model shapes: `models.py` and `layers.py` — mismatched dimensions in `edges_layer` / `nodes_layer` are common causes of runtime reshape errors.
- Data interface: `data_loader.py` and `data/sparse_molecular_dataset.py` — changing dataset paths or formats should be done here.
- Metrics & reward: `utils.py` (MolecularMetrics) — if a metric suddenly breaks (e.g., RDKit sanitization), follow the `_avoid_sanitization_error` pattern already used.

## Example quick tasks & how to approach them
- Add a new quantum ansatz parameterization: edit `gen_circuit` in `main.py` (or the patched scripts), ensure returned vector length equals `config.qubits` (z-dim), and update any `gen_weights` creation (see main: `torch.tensor(list(np.random.rand(...)))`).
- Debug a shape mismatch: print shapes before `self.G(z)` in `main.py` / `solver.py`, confirm `z.shape == (batch_size, z_dim)` and that `Generator` z_dim aligns with first layer.

## Files to inspect first (high signal)
- `README.md` — run examples and dependency hints.
- `main.py`, `p2_qgan_hg.py`, `p4_qgan_hg.py` — entrypoints with quantum code.
- `solver.py` — training loop, model creation, postprocessing, reward, saving logic.
- `models.py`, `layers.py` — network definitions and graph conv building blocks.
- `utils.py` — metrics (MolecularMetrics), IO helpers, and dataset utilities.

## Final notes for the agent
- Favor minimal, localized edits: change the circuit in the qnode, the `postprocess` choice, or dataset path first.
- Preserve checkpoint naming and CSV append behaviors unless intentionally changing experiment logging semantics.
- If adding tests, target small CPU-only runs (`num_iters=1`, `batch_size=1`) and mock RDKit-heavy functions where possible.

Please review — tell me if you'd like more detail on any section (examples, exact code links, or a short checklist for contributing changes).
