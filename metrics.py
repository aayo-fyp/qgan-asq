"""
Molecular Metrics Module for HQ-Cycle-MolGAN.

This module provides centralized implementations of molecular quality metrics
used for evaluating generated molecules in drug discovery applications.

Metrics implemented:
- QED (Quantitative Estimation of Druglikeness)
- LogP (Water-Octanol Partition Coefficient)
- SA Score (Synthetic Accessibility)
- NP Score (Natural Product-likeness)
- Validity, Novelty, Uniqueness, Diversity

References:
- QED: Bickerton et al., Nature Chemistry, 2012
- SA Score: Ertl & Schuffenhauer, J. Cheminform., 2009
- NP Score: Ertl et al., J. Chem. Inf. Model., 2007
"""

import math
import pickle
import gzip
from typing import List, Optional, Set, Dict, Union
import numpy as np

from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import QED as RDKitQED
from rdkit.Chem import Crippen
from rdkit.Chem import rdMolDescriptors


# =============================================================================
# Model Loading (SA and NP score models)
# =============================================================================

def _load_np_model(path: str = 'data/NP_score.pkl.gz') -> dict:
    """Load Natural Product score model."""
    try:
        return pickle.load(gzip.open(path))
    except Exception:
        return {}


def _load_sa_model(path: str = 'data/SA_score.pkl.gz') -> dict:
    """Load Synthetic Accessibility score model."""
    try:
        data = pickle.load(gzip.open(path))
        return {i[j]: float(i[0]) for i in data for j in range(1, len(i))}
    except Exception:
        return {}


# Global model instances (loaded once)
_NP_MODEL = None
_SA_MODEL = None


def _get_np_model() -> dict:
    """Get or load NP model (lazy loading)."""
    global _NP_MODEL
    if _NP_MODEL is None:
        _NP_MODEL = _load_np_model()
    return _NP_MODEL


def _get_sa_model() -> dict:
    """Get or load SA model (lazy loading)."""
    global _SA_MODEL
    if _SA_MODEL is None:
        _SA_MODEL = _load_sa_model()
    return _SA_MODEL


# =============================================================================
# Utility Functions
# =============================================================================

def _safe_op(func, default=None):
    """Execute function safely, returning default on error."""
    try:
        return func()
    except Exception:
        return default


def normalize(x: np.ndarray, x_min: float, x_max: float) -> np.ndarray:
    """Normalize values to [0, 1] range."""
    return np.clip((x - x_min) / (x_max - x_min), 0.0, 1.0)


def is_valid_mol(mol) -> bool:
    """Check if molecule is valid (not None and has valid SMILES)."""
    if mol is None:
        return False
    try:
        smiles = Chem.MolToSmiles(mol)
        return smiles != '' and '*' not in smiles and '.' not in smiles
    except Exception:
        return False


def mol_to_smiles(mol) -> Optional[str]:
    """Convert molecule to SMILES string safely."""
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


# =============================================================================
# Individual Metric Functions
# =============================================================================

def qed_score(mol) -> float:
    """
    Compute Quantitative Estimation of Druglikeness (QED) score.
    
    QED is a composite metric that evaluates a molecule's overall druglikeness
    based on molecular weight, LogP, H-bond donors/acceptors, PSA, rotatable bonds,
    aromatic rings, and structural alerts.
    
    Args:
        mol: RDKit molecule object.
        
    Returns:
        QED score in range [0, 1]. Higher is more drug-like.
        Returns 0.0 if molecule is invalid.
    """
    if mol is None:
        return 0.0
    return _safe_op(lambda: RDKitQED.qed(mol), default=0.0)


def logp_score(mol, normalize_score: bool = False) -> float:
    """
    Compute Crippen LogP (water-octanol partition coefficient).
    
    LogP measures lipophilicity - the balance between hydrophilicity and 
    lipophilicity. Optimal drug-like LogP is typically in range [-0.4, 5.6].
    
    Args:
        mol: RDKit molecule object.
        normalize_score: If True, normalize to [0, 1] range.
        
    Returns:
        LogP score. If normalized, values are clipped to [0, 1].
        Returns -3.0 (or 0.0 if normalized) for invalid molecules.
    """
    if mol is None:
        return 0.0 if normalize_score else -3.0
    
    score = _safe_op(lambda: Crippen.MolLogP(mol), default=-3.0)
    
    if normalize_score:
        # Normalization range from paper: [-2.12, 6.04]
        return float(normalize(np.array([score]), -2.12178879609, 6.0429063424)[0])
    return score


def sa_score(mol, normalize_score: bool = False) -> float:
    """
    Compute Synthetic Accessibility (SA) score.
    
    SA score quantifies how difficult a molecule is to synthesize.
    Range is [1, 10] where 1 = easy to synthesize, 10 = very difficult.
    
    Based on Ertl & Schuffenhauer, J. Cheminform., 2009.
    
    Args:
        mol: RDKit molecule object.
        normalize_score: If True, normalize to [0, 1] where 1 = easy.
        
    Returns:
        SA score. Raw: [1, 10], Normalized: [0, 1] (inverted, higher = easier).
        Returns 10.0 (or 0.0 if normalized) for invalid molecules.
    """
    if mol is None:
        return 0.0 if normalize_score else 10.0
    
    sa_model = _get_sa_model()
    if not sa_model:
        return 0.0 if normalize_score else 10.0
    
    try:
        # Get Morgan fingerprint
        fp = rdMolDescriptors.GetMorganFingerprint(mol, 2)
        fps = fp.GetNonzeroElements()
        
        # Fragment score
        score1 = 0.0
        nf = 0
        for bit_id, v in fps.items():
            nf += v
            score1 += sa_model.get(bit_id, -4) * v
        score1 /= nf if nf > 0 else 1
        
        # Complexity features
        n_atoms = mol.GetNumAtoms()
        n_chiral = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
        ri = mol.GetRingInfo()
        n_spiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
        n_bridgeheads = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
        n_macrocycles = sum(1 for x in ri.AtomRings() if len(x) > 8)
        
        # Penalties
        size_penalty = n_atoms ** 1.005 - n_atoms
        stereo_penalty = math.log10(n_chiral + 1)
        spiro_penalty = math.log10(n_spiro + 1)
        bridge_penalty = math.log10(n_bridgeheads + 1)
        macrocycle_penalty = math.log10(2) if n_macrocycles > 0 else 0.0
        
        score2 = -size_penalty - stereo_penalty - spiro_penalty - bridge_penalty - macrocycle_penalty
        
        # Fingerprint density correction
        score3 = math.log(float(n_atoms) / len(fps)) * 0.5 if n_atoms > len(fps) else 0.0
        
        # Combine and scale to [1, 10]
        sascore = score1 + score2 + score3
        sascore = 11.0 - (sascore - (-4.0) + 1) / (2.5 - (-4.0)) * 9.0
        
        # Smoothing
        if sascore > 8.0:
            sascore = 8.0 + math.log(sascore + 1.0 - 9.0)
        sascore = max(1.0, min(10.0, sascore))
        
    except Exception:
        sascore = 10.0
    
    if normalize_score:
        # Normalize: [5, 1.5] -> [0, 1] (inverted so higher = easier)
        return float(normalize(np.array([sascore]), 5, 1.5)[0])
    return sascore


def np_score(mol, normalize_score: bool = False) -> float:
    """
    Compute Natural Product-likeness (NP) score.
    
    NP score measures how similar a molecule is to natural products.
    Based on Ertl et al., J. Chem. Inf. Model., 2007.
    
    Args:
        mol: RDKit molecule object.
        normalize_score: If True, normalize to [0, 1] range.
        
    Returns:
        NP score. Higher values indicate more natural product-like.
        Returns -4.0 (or 0.0 if normalized) for invalid molecules.
    """
    if mol is None:
        return 0.0 if normalize_score else -4.0
    
    np_model = _get_np_model()
    if not np_model:
        return 0.0 if normalize_score else -4.0
    
    try:
        fp = rdMolDescriptors.GetMorganFingerprint(mol, 2)
        bits = fp.GetNonzeroElements()
        score = sum(np_model.get(bit, 0) for bit in bits) / float(mol.GetNumAtoms())
        
        # Prevent score explosion
        if score > 4:
            score = 4 + math.log10(score - 4 + 1)
        elif score < -4:
            score = -4 - math.log10(-4 - score + 1)
            
    except Exception:
        score = -4.0
    
    if normalize_score:
        return float(normalize(np.array([score]), -3, 1)[0])
    return score


# =============================================================================
# Batch Metric Functions
# =============================================================================

def qed_scores(mols: List, normalize_score: bool = False) -> np.ndarray:
    """Compute QED scores for a batch of molecules."""
    return np.array([qed_score(mol) for mol in mols], dtype=np.float32)


def logp_scores(mols: List, normalize_score: bool = False) -> np.ndarray:
    """Compute LogP scores for a batch of molecules."""
    return np.array([logp_score(mol, normalize_score) for mol in mols], dtype=np.float32)


def sa_scores(mols: List, normalize_score: bool = False) -> np.ndarray:
    """Compute SA scores for a batch of molecules."""
    return np.array([sa_score(mol, normalize_score) for mol in mols], dtype=np.float32)


def np_scores(mols: List, normalize_score: bool = False) -> np.ndarray:
    """Compute NP scores for a batch of molecules."""
    return np.array([np_score(mol, normalize_score) for mol in mols], dtype=np.float32)


def validity_scores(mols: List) -> np.ndarray:
    """
    Compute per-molecule validity scores (0 or 1).
    
    Args:
        mols: List of RDKit molecule objects.
        
    Returns:
        Array of 0s and 1s indicating validity.
    """
    return np.array([1.0 if is_valid_mol(mol) else 0.0 for mol in mols], dtype=np.float32)


def validity_ratio(mols: List) -> float:
    """
    Compute fraction of valid molecules.
    
    Args:
        mols: List of RDKit molecule objects.
        
    Returns:
        Fraction in [0, 1].
    """
    if not mols:
        return 0.0
    return float(validity_scores(mols).mean())


def uniqueness_scores(mols: List) -> np.ndarray:
    """
    Compute per-molecule uniqueness scores.
    
    Score is based on how many times each SMILES appears in the batch.
    Unique molecules get score 1.0, duplicates get lower scores.
    
    Args:
        mols: List of RDKit molecule objects.
        
    Returns:
        Array of uniqueness scores in [0, 1].
    """
    smiles_list = [mol_to_smiles(mol) if is_valid_mol(mol) else '' for mol in mols]
    scores = []
    for s in smiles_list:
        if s == '':
            scores.append(0.0)
        else:
            count = smiles_list.count(s)
            scores.append(min(1.0, 0.75 + 1.0 / count))
    return np.clip(np.array(scores, dtype=np.float32), 0.0, 1.0)


def uniqueness_ratio(mols: List) -> float:
    """
    Compute fraction of unique valid molecules.
    
    Args:
        mols: List of RDKit molecule objects.
        
    Returns:
        Fraction of unique molecules among valid ones.
    """
    valid_mols = [mol for mol in mols if is_valid_mol(mol)]
    if not valid_mols:
        return 0.0
    unique_smiles = set(mol_to_smiles(mol) for mol in valid_mols)
    return len(unique_smiles) / len(valid_mols)


def novelty_scores(mols: List, train_smiles: Set[str]) -> np.ndarray:
    """
    Compute per-molecule novelty scores.
    
    A molecule is novel if it's valid and its SMILES is not in the training set.
    
    Args:
        mols: List of RDKit molecule objects.
        train_smiles: Set of SMILES strings from training data.
        
    Returns:
        Array of 0s and 1s indicating novelty.
    """
    scores = []
    for mol in mols:
        if not is_valid_mol(mol):
            scores.append(0.0)
        else:
            smiles = mol_to_smiles(mol)
            scores.append(1.0 if smiles not in train_smiles else 0.0)
    return np.array(scores, dtype=np.float32)


def novelty_ratio(mols: List, train_smiles: Set[str]) -> float:
    """
    Compute fraction of novel molecules among valid ones.
    
    Args:
        mols: List of RDKit molecule objects.
        train_smiles: Set of SMILES strings from training data.
        
    Returns:
        Fraction of novel molecules.
    """
    valid_mols = [mol for mol in mols if is_valid_mol(mol)]
    if not valid_mols:
        return 0.0
    novel_count = sum(1 for mol in valid_mols if mol_to_smiles(mol) not in train_smiles)
    return novel_count / len(valid_mols)


def diversity_scores(mols: List, reference_mols: List = None, n_reference: int = 100) -> np.ndarray:
    """
    Compute per-molecule diversity scores based on Tanimoto distance.
    
    Diversity is measured as average Tanimoto distance to a set of reference molecules.
    
    Args:
        mols: List of RDKit molecule objects to evaluate.
        reference_mols: Reference molecules for comparison. If None, uses mols itself.
        n_reference: Number of reference molecules to sample.
        
    Returns:
        Array of diversity scores in [0, 1].
    """
    if reference_mols is None:
        reference_mols = mols
    
    # Sample reference molecules
    if len(reference_mols) > n_reference:
        indices = np.random.choice(len(reference_mols), n_reference, replace=False)
        reference_mols = [reference_mols[i] for i in indices]
    
    # Compute reference fingerprints
    ref_fps = []
    for mol in reference_mols:
        if mol is not None:
            try:
                fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 4, nBits=2048)
                ref_fps.append(fp)
            except Exception:
                pass
    
    if not ref_fps:
        return np.zeros(len(mols), dtype=np.float32)
    
    # Compute diversity for each molecule
    scores = []
    for mol in mols:
        if mol is None:
            scores.append(0.0)
            continue
        try:
            mol_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 4, nBits=2048)
            distances = DataStructs.BulkTanimotoSimilarity(mol_fp, ref_fps, returnDistance=True)
            scores.append(float(np.mean(distances)))
        except Exception:
            scores.append(0.0)
    
    # Normalize to [0, 1] using empirical range
    scores = np.array(scores, dtype=np.float32)
    return normalize(scores, 0.9, 0.945)


def diversity_ratio(mols: List) -> float:
    """
    Compute overall diversity of a set of molecules.
    
    Args:
        mols: List of RDKit molecule objects.
        
    Returns:
        Average pairwise Tanimoto distance.
    """
    valid_mols = [mol for mol in mols if is_valid_mol(mol)]
    if len(valid_mols) < 2:
        return 0.0
    
    scores = diversity_scores(valid_mols, valid_mols)
    return float(scores.mean())


# =============================================================================
# Combined Reward Functions
# =============================================================================

def compute_reward_batch(
    mols: List,
    metrics: List[str] = None,
    train_smiles: Set[str] = None,
    normalize_all: bool = True
) -> np.ndarray:
    """
    Compute combined reward for a batch of molecules.
    
    This is the main function used during GAN training to compute rewards
    for the generator.
    
    Args:
        mols: List of RDKit molecule objects.
        metrics: List of metrics to include. Options: 'qed', 'logp', 'sa', 'np',
                 'validity', 'uniqueness', 'novelty', 'diversity'.
                 Default: ['validity', 'sa'].
        train_smiles: Set of training SMILES (required for novelty).
        normalize_all: Whether to normalize all metrics to [0, 1].
        
    Returns:
        Array of shape (n_mols, 1) with combined reward scores.
    """
    if metrics is None:
        metrics = ['validity', 'sa']
    
    reward = np.ones(len(mols), dtype=np.float32)
    
    for metric in metrics:
        metric = metric.lower()
        
        if metric == 'qed':
            reward *= qed_scores(mols)
        elif metric == 'logp':
            reward *= logp_scores(mols, normalize_score=normalize_all)
        elif metric == 'sa' or metric == 'sas':
            reward *= sa_scores(mols, normalize_score=normalize_all)
        elif metric == 'np':
            reward *= np_scores(mols, normalize_score=normalize_all)
        elif metric == 'validity':
            reward *= validity_scores(mols)
        elif metric == 'unique' or metric == 'uniqueness':
            reward *= uniqueness_scores(mols)
        elif metric == 'novelty' or metric == 'novel':
            if train_smiles is None:
                raise ValueError("train_smiles required for novelty metric")
            reward *= novelty_scores(mols, train_smiles)
        elif metric == 'diversity':
            reward *= diversity_scores(mols)
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    return reward.reshape(-1, 1)


def compute_all_metrics(
    mols: List,
    train_smiles: Set[str] = None,
    reference_mols: List = None,
    normalize_all: bool = False
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Compute all available metrics for a batch of molecules.
    
    Args:
        mols: List of RDKit molecule objects.
        train_smiles: Set of training SMILES (for novelty).
        reference_mols: Reference molecules (for diversity).
        normalize_all: Whether to normalize scores.
        
    Returns:
        Dictionary with metric names as keys and values as either:
        - Per-molecule arrays (for *_scores)
        - Single floats (for *_ratio)
    """
    results = {
        # Per-molecule scores
        'qed_scores': qed_scores(mols),
        'logp_scores': logp_scores(mols, normalize_score=normalize_all),
        'sa_scores': sa_scores(mols, normalize_score=normalize_all),
        'np_scores': np_scores(mols, normalize_score=normalize_all),
        'validity_scores': validity_scores(mols),
        'uniqueness_scores': uniqueness_scores(mols),
        
        # Aggregate ratios
        'validity_ratio': validity_ratio(mols),
        'uniqueness_ratio': uniqueness_ratio(mols),
        'diversity_ratio': diversity_ratio(mols),
        
        # Mean scores (excluding invalid)
        'qed_mean': float(np.nanmean([s for s, v in zip(qed_scores(mols), validity_scores(mols)) if v > 0]) or 0),
        'logp_mean': float(np.nanmean([s for s, v in zip(logp_scores(mols), validity_scores(mols)) if v > 0]) or 0),
        'sa_mean': float(np.nanmean([s for s, v in zip(sa_scores(mols), validity_scores(mols)) if v > 0]) or 0),
        'np_mean': float(np.nanmean([s for s, v in zip(np_scores(mols), validity_scores(mols)) if v > 0]) or 0),
    }
    
    # Add novelty if train_smiles provided
    if train_smiles is not None:
        results['novelty_scores'] = novelty_scores(mols, train_smiles)
        results['novelty_ratio'] = novelty_ratio(mols, train_smiles)
    
    # Add diversity scores
    results['diversity_scores'] = diversity_scores(mols, reference_mols)
    
    return results


# =============================================================================
# Drug Candidate Score (Combined Metric)
# =============================================================================

def _constant_bump(x: np.ndarray, x_low: float, x_high: float, decay: float = 0.025) -> np.ndarray:
    """Apply constant bump function for drugcandidate score."""
    return np.select(
        condlist=[x <= x_low, x >= x_high],
        choicelist=[
            np.exp(-(x - x_low) ** 2 / decay),
            np.exp(-(x - x_high) ** 2 / decay)
        ],
        default=np.ones_like(x)
    )


def drugcandidate_scores(mols: List, train_smiles: Set[str] = None) -> np.ndarray:
    """
    Compute drug candidate scores combining multiple metrics.
    
    This is a composite score that combines LogP, SA, and novelty.
    
    Args:
        mols: List of RDKit molecule objects.
        train_smiles: Set of training SMILES (for novelty component).
        
    Returns:
        Array of drug candidate scores in [0, 1].
    """
    logp = logp_scores(mols, normalize_score=True)
    sa = sa_scores(mols, normalize_score=True)
    
    if train_smiles is not None:
        novel = novelty_scores(mols, train_smiles)
    else:
        novel = np.ones(len(mols), dtype=np.float32)
    
    # Combine with constant bump on LogP
    scores = (
        _constant_bump(logp, 0.210, 0.945) +
        sa +
        novel +
        (1 - novel) * 0.3
    ) / 4
    
    return scores


# =============================================================================
# Compatibility Layer (for existing code using MolecularMetrics class)
# =============================================================================

class MolecularMetrics:
    """
    Compatibility class that wraps the functional API.
    
    This maintains backward compatibility with code using the old
    MolecularMetrics class from utils.py.
    """
    
    @staticmethod
    def valid_scores(mols):
        return validity_scores(mols)
    
    @staticmethod
    def valid_total_score(mols):
        return validity_ratio(mols)
    
    @staticmethod
    def valid_filter(mols):
        return [mol for mol in mols if is_valid_mol(mol)]
    
    @staticmethod
    def unique_scores(mols):
        return uniqueness_scores(mols)
    
    @staticmethod
    def unique_total_score(mols):
        return uniqueness_ratio(mols)
    
    @staticmethod
    def novel_scores(mols, data):
        train_smiles = set(data.smiles) if hasattr(data, 'smiles') else set()
        return novelty_scores(mols, train_smiles)
    
    @staticmethod
    def novel_total_score(mols, data):
        train_smiles = set(data.smiles) if hasattr(data, 'smiles') else set()
        return novelty_ratio(mols, train_smiles)
    
    @staticmethod
    def quantitative_estimation_druglikeness_scores(mols, norm=False):
        return qed_scores(mols)
    
    @staticmethod
    def water_octanol_partition_coefficient_scores(mols, norm=False):
        return logp_scores(mols, normalize_score=norm)
    
    @staticmethod
    def synthetic_accessibility_score_scores(mols, norm=False):
        return sa_scores(mols, normalize_score=norm)
    
    @staticmethod
    def natural_product_scores(mols, norm=False):
        return np_scores(mols, normalize_score=norm)
    
    @staticmethod
    def diversity_scores(mols, data):
        reference = data.data if hasattr(data, 'data') else None
        return diversity_scores(mols, reference)
    
    @staticmethod
    def drugcandidate_scores(mols, data):
        train_smiles = set(data.smiles) if hasattr(data, 'smiles') else None
        return drugcandidate_scores(mols, train_smiles)


# =============================================================================
# Convenience function matching utils.py all_scores
# =============================================================================

def all_scores(mols: List, data, norm: bool = False) -> tuple:
    """
    Compute all scores for a batch of molecules.
    
    This function matches the signature of utils.all_scores for compatibility.
    
    Args:
        mols: List of RDKit molecule objects.
        data: Dataset object with 'smiles' and 'data' attributes.
        norm: Whether to normalize scores.
        
    Returns:
        Tuple of (per_mol_metrics_dict, aggregate_metrics_dict).
    """
    train_smiles = set(data.smiles) if hasattr(data, 'smiles') else set()
    reference = data.data if hasattr(data, 'data') else None
    
    # Per-molecule metrics (filter out None values for display)
    m0 = {
        'NP score': list(filter(lambda x: x is not None, np_scores(mols, normalize_score=norm))),
        'QED score': list(filter(lambda x: x is not None, qed_scores(mols))),
        'logP score': list(filter(lambda x: x is not None, logp_scores(mols, normalize_score=norm))),
        'SA score': list(filter(lambda x: x is not None, sa_scores(mols, normalize_score=norm))),
        'diversity score': list(filter(lambda x: x is not None, diversity_scores(mols, reference))),
        'drugcandidate score': list(filter(lambda x: x is not None, drugcandidate_scores(mols, train_smiles))),
    }
    
    # Aggregate metrics (percentages)
    m1 = {
        'valid score': validity_ratio(mols) * 100,
        'unique score': uniqueness_ratio(mols) * 100,
        'novel score': novelty_ratio(mols, train_smiles) * 100,
    }
    
    return m0, m1

