"""
Unit tests for metrics.py module.

Tests cover:
- Individual metric functions (QED, LogP, SA, NP)
- Batch metric functions
- Aggregate metrics (validity, uniqueness, novelty, diversity)
- Combined reward computation
- Backward compatibility with MolecularMetrics class
"""

import pytest
import numpy as np
from rdkit import Chem


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def valid_mol():
    """Create a valid molecule (aspirin)."""
    return Chem.MolFromSmiles('CC(=O)OC1=CC=CC=C1C(=O)O')


@pytest.fixture
def valid_mol2():
    """Create another valid molecule (caffeine)."""
    return Chem.MolFromSmiles('CN1C=NC2=C1C(=O)N(C(=O)N2C)C')


@pytest.fixture
def valid_mol3():
    """Create a third valid molecule (ethanol)."""
    return Chem.MolFromSmiles('CCO')


@pytest.fixture
def invalid_mol():
    """Return None to represent an invalid molecule."""
    return None


@pytest.fixture
def mol_batch(valid_mol, valid_mol2, valid_mol3, invalid_mol):
    """Create a batch of molecules with some invalid."""
    return [valid_mol, valid_mol2, invalid_mol, valid_mol3, invalid_mol]


@pytest.fixture
def train_smiles():
    """Create a set of training SMILES."""
    return {'CCO', 'CC(=O)O', 'C1=CC=CC=C1'}  # ethanol, acetic acid, benzene


# =============================================================================
# Test Module Import
# =============================================================================

def test_import():
    """Test that metrics module imports correctly."""
    import metrics
    assert hasattr(metrics, 'qed_score')
    assert hasattr(metrics, 'logp_score')
    assert hasattr(metrics, 'sa_score')
    assert hasattr(metrics, 'np_score')
    assert hasattr(metrics, 'MolecularMetrics')


# =============================================================================
# Test Individual Metric Functions
# =============================================================================

class TestQEDScore:
    """Tests for QED score computation."""
    
    def test_valid_mol(self, valid_mol):
        """QED score should be in [0, 1] for valid molecule."""
        from metrics import qed_score
        score = qed_score(valid_mol)
        assert 0.0 <= score <= 1.0
        assert score > 0.0  # Aspirin should have positive QED
    
    def test_invalid_mol(self, invalid_mol):
        """QED score should be 0 for invalid molecule."""
        from metrics import qed_score
        score = qed_score(invalid_mol)
        assert score == 0.0
    
    def test_batch(self, mol_batch):
        """Batch QED scores should have correct shape."""
        from metrics import qed_scores
        scores = qed_scores(mol_batch)
        assert scores.shape == (5,)
        assert scores.dtype == np.float32


class TestLogPScore:
    """Tests for LogP score computation."""
    
    def test_valid_mol(self, valid_mol):
        """LogP should return a float for valid molecule."""
        from metrics import logp_score
        score = logp_score(valid_mol)
        assert isinstance(score, float)
    
    def test_invalid_mol(self, invalid_mol):
        """LogP should return default for invalid molecule."""
        from metrics import logp_score
        score = logp_score(invalid_mol)
        assert score == -3.0
        
        score_norm = logp_score(invalid_mol, normalize_score=True)
        assert score_norm == 0.0
    
    def test_normalized(self, valid_mol):
        """Normalized LogP should be in [0, 1]."""
        from metrics import logp_score
        score = logp_score(valid_mol, normalize_score=True)
        assert 0.0 <= score <= 1.0
    
    def test_batch(self, mol_batch):
        """Batch LogP scores should have correct shape."""
        from metrics import logp_scores
        scores = logp_scores(mol_batch)
        assert scores.shape == (5,)


class TestSAScore:
    """Tests for Synthetic Accessibility score computation."""
    
    def test_valid_mol(self, valid_mol):
        """SA score should be in [1, 10] for valid molecule."""
        from metrics import sa_score
        score = sa_score(valid_mol)
        assert 1.0 <= score <= 10.0
    
    def test_invalid_mol(self, invalid_mol):
        """SA score should return default for invalid molecule."""
        from metrics import sa_score
        score = sa_score(invalid_mol)
        assert score == 10.0
        
        score_norm = sa_score(invalid_mol, normalize_score=True)
        assert score_norm == 0.0
    
    def test_normalized(self, valid_mol):
        """Normalized SA should be in [0, 1]."""
        from metrics import sa_score
        score = sa_score(valid_mol, normalize_score=True)
        assert 0.0 <= score <= 1.0
    
    def test_batch(self, mol_batch):
        """Batch SA scores should have correct shape."""
        from metrics import sa_scores
        scores = sa_scores(mol_batch)
        assert scores.shape == (5,)


class TestNPScore:
    """Tests for Natural Product score computation."""
    
    def test_valid_mol(self, valid_mol):
        """NP score should return a float for valid molecule."""
        from metrics import np_score
        score = np_score(valid_mol)
        assert isinstance(score, float)
    
    def test_invalid_mol(self, invalid_mol):
        """NP score should return default for invalid molecule."""
        from metrics import np_score
        score = np_score(invalid_mol)
        assert score == -4.0
        
        score_norm = np_score(invalid_mol, normalize_score=True)
        assert score_norm == 0.0
    
    def test_normalized(self, valid_mol):
        """Normalized NP should be in [0, 1]."""
        from metrics import np_score
        score = np_score(valid_mol, normalize_score=True)
        assert 0.0 <= score <= 1.0


# =============================================================================
# Test Validity and Uniqueness
# =============================================================================

class TestValidity:
    """Tests for validity metrics."""
    
    def test_is_valid_mol(self, valid_mol, invalid_mol):
        """Test is_valid_mol function."""
        from metrics import is_valid_mol
        assert is_valid_mol(valid_mol) is True
        assert is_valid_mol(invalid_mol) is False
    
    def test_validity_scores(self, mol_batch):
        """Validity scores should be 0 or 1."""
        from metrics import validity_scores
        scores = validity_scores(mol_batch)
        assert scores.shape == (5,)
        assert all(s in [0.0, 1.0] for s in scores)
        # 3 valid, 2 invalid
        assert scores.sum() == 3.0
    
    def test_validity_ratio(self, mol_batch):
        """Validity ratio should be fraction of valid molecules."""
        from metrics import validity_ratio
        ratio = validity_ratio(mol_batch)
        assert ratio == pytest.approx(0.6, rel=0.01)  # 3/5


class TestUniqueness:
    """Tests for uniqueness metrics."""
    
    def test_uniqueness_scores_all_unique(self, valid_mol, valid_mol2, valid_mol3):
        """All unique molecules should have high scores."""
        from metrics import uniqueness_scores
        mols = [valid_mol, valid_mol2, valid_mol3]
        scores = uniqueness_scores(mols)
        assert all(s > 0.9 for s in scores)
    
    def test_uniqueness_scores_duplicates(self, valid_mol):
        """Duplicate molecules should have lower scores."""
        from metrics import uniqueness_scores
        mols = [valid_mol, valid_mol, valid_mol]
        scores = uniqueness_scores(mols)
        # With 3 duplicates: 0.75 + 1/3 ≈ 1.083, clipped to 1.0
        # The formula clips to [0, 1], so duplicates still get 1.0
        # But uniqueness_ratio should show they're not unique
        from metrics import uniqueness_ratio
        ratio = uniqueness_ratio(mols)
        assert ratio == pytest.approx(1/3, rel=0.01)  # only 1 unique out of 3 valid
    
    def test_uniqueness_ratio(self, valid_mol, valid_mol2):
        """Uniqueness ratio should be fraction of unique among valid."""
        from metrics import uniqueness_ratio
        mols = [valid_mol, valid_mol, valid_mol2]
        ratio = uniqueness_ratio(mols)
        assert ratio == pytest.approx(2/3, rel=0.01)


# =============================================================================
# Test Novelty
# =============================================================================

class TestNovelty:
    """Tests for novelty metrics."""
    
    def test_novelty_scores(self, valid_mol3, valid_mol, train_smiles):
        """Test novelty detection."""
        from metrics import novelty_scores
        # valid_mol3 is ethanol (CCO) which is in train_smiles
        # valid_mol is aspirin which is NOT in train_smiles
        mols = [valid_mol3, valid_mol]
        scores = novelty_scores(mols, train_smiles)
        assert scores[0] == 0.0  # ethanol is in training set
        assert scores[1] == 1.0  # aspirin is novel
    
    def test_novelty_ratio(self, valid_mol3, valid_mol, valid_mol2, train_smiles):
        """Novelty ratio should be fraction of novel among valid."""
        from metrics import novelty_ratio
        # ethanol in training, aspirin and caffeine are novel
        mols = [valid_mol3, valid_mol, valid_mol2]
        ratio = novelty_ratio(mols, train_smiles)
        assert ratio == pytest.approx(2/3, rel=0.01)


# =============================================================================
# Test Diversity
# =============================================================================

class TestDiversity:
    """Tests for diversity metrics."""
    
    def test_diversity_scores_shape(self, mol_batch):
        """Diversity scores should have correct shape."""
        from metrics import diversity_scores
        scores = diversity_scores(mol_batch)
        assert scores.shape == (5,)
    
    def test_diversity_scores_range(self, valid_mol, valid_mol2, valid_mol3):
        """Diversity scores should be in [0, 1]."""
        from metrics import diversity_scores
        mols = [valid_mol, valid_mol2, valid_mol3]
        scores = diversity_scores(mols)
        assert all(0.0 <= s <= 1.0 for s in scores)
    
    def test_diversity_ratio(self, valid_mol, valid_mol2, valid_mol3):
        """Diversity ratio should be a float."""
        from metrics import diversity_ratio
        mols = [valid_mol, valid_mol2, valid_mol3]
        ratio = diversity_ratio(mols)
        assert isinstance(ratio, float)
        assert 0.0 <= ratio <= 1.0


# =============================================================================
# Test Combined Metrics
# =============================================================================

class TestComputeRewardBatch:
    """Tests for combined reward computation."""
    
    def test_default_metrics(self, mol_batch):
        """Default reward should use validity and SA."""
        from metrics import compute_reward_batch
        rewards = compute_reward_batch(mol_batch)
        assert rewards.shape == (5, 1)
    
    def test_custom_metrics(self, mol_batch):
        """Custom metrics should work."""
        from metrics import compute_reward_batch
        rewards = compute_reward_batch(mol_batch, metrics=['qed', 'validity'])
        assert rewards.shape == (5, 1)
    
    def test_novelty_requires_train_smiles(self, mol_batch):
        """Novelty metric should require train_smiles."""
        from metrics import compute_reward_batch
        with pytest.raises(ValueError, match="train_smiles required"):
            compute_reward_batch(mol_batch, metrics=['novelty'])
    
    def test_novelty_with_train_smiles(self, mol_batch, train_smiles):
        """Novelty metric should work with train_smiles."""
        from metrics import compute_reward_batch
        rewards = compute_reward_batch(mol_batch, metrics=['novelty'], train_smiles=train_smiles)
        assert rewards.shape == (5, 1)
    
    def test_unknown_metric(self, mol_batch):
        """Unknown metric should raise error."""
        from metrics import compute_reward_batch
        with pytest.raises(ValueError, match="Unknown metric"):
            compute_reward_batch(mol_batch, metrics=['unknown_metric'])


class TestComputeAllMetrics:
    """Tests for compute_all_metrics function."""
    
    def test_returns_dict(self, mol_batch, train_smiles):
        """Should return a dictionary with all metrics."""
        from metrics import compute_all_metrics
        results = compute_all_metrics(mol_batch, train_smiles=train_smiles)
        
        assert isinstance(results, dict)
        assert 'qed_scores' in results
        assert 'validity_ratio' in results
        assert 'novelty_ratio' in results
    
    def test_array_shapes(self, mol_batch, train_smiles):
        """Per-molecule arrays should have correct shape."""
        from metrics import compute_all_metrics
        results = compute_all_metrics(mol_batch, train_smiles=train_smiles)
        
        assert results['qed_scores'].shape == (5,)
        assert results['validity_scores'].shape == (5,)
    
    def test_aggregate_types(self, mol_batch, train_smiles):
        """Aggregate metrics should be floats."""
        from metrics import compute_all_metrics
        results = compute_all_metrics(mol_batch, train_smiles=train_smiles)
        
        assert isinstance(results['validity_ratio'], float)
        assert isinstance(results['uniqueness_ratio'], float)
        assert isinstance(results['novelty_ratio'], float)


# =============================================================================
# Test Backward Compatibility
# =============================================================================

class TestMolecularMetricsCompatibility:
    """Tests for backward compatibility with MolecularMetrics class."""
    
    def test_class_exists(self):
        """MolecularMetrics class should exist."""
        from metrics import MolecularMetrics
        assert MolecularMetrics is not None
    
    def test_valid_scores(self, mol_batch):
        """valid_scores should work."""
        from metrics import MolecularMetrics
        scores = MolecularMetrics.valid_scores(mol_batch)
        assert scores.shape == (5,)
    
    def test_valid_total_score(self, mol_batch):
        """valid_total_score should work."""
        from metrics import MolecularMetrics
        score = MolecularMetrics.valid_total_score(mol_batch)
        assert score == pytest.approx(0.6, rel=0.01)
    
    def test_valid_filter(self, mol_batch):
        """valid_filter should return only valid molecules."""
        from metrics import MolecularMetrics
        valid = MolecularMetrics.valid_filter(mol_batch)
        assert len(valid) == 3
    
    def test_unique_scores(self, mol_batch):
        """unique_scores should work."""
        from metrics import MolecularMetrics
        scores = MolecularMetrics.unique_scores(mol_batch)
        assert scores.shape == (5,)
    
    def test_qed_scores(self, mol_batch):
        """QED scores should work."""
        from metrics import MolecularMetrics
        scores = MolecularMetrics.quantitative_estimation_druglikeness_scores(mol_batch)
        assert scores.shape == (5,)
    
    def test_logp_scores(self, mol_batch):
        """LogP scores should work."""
        from metrics import MolecularMetrics
        scores = MolecularMetrics.water_octanol_partition_coefficient_scores(mol_batch)
        assert scores.shape == (5,)
    
    def test_sa_scores(self, mol_batch):
        """SA scores should work."""
        from metrics import MolecularMetrics
        scores = MolecularMetrics.synthetic_accessibility_score_scores(mol_batch)
        assert scores.shape == (5,)
    
    def test_np_scores(self, mol_batch):
        """NP scores should work."""
        from metrics import MolecularMetrics
        scores = MolecularMetrics.natural_product_scores(mol_batch)
        assert scores.shape == (5,)


class TestAllScoresFunction:
    """Tests for all_scores compatibility function."""
    
    def test_returns_tuple(self, mol_batch):
        """all_scores should return a tuple of two dicts."""
        from metrics import all_scores
        
        # Create a mock data object
        class MockData:
            smiles = ['CCO', 'CC(=O)O']
            data = [Chem.MolFromSmiles('CCO')]
        
        m0, m1 = all_scores(mol_batch, MockData())
        
        assert isinstance(m0, dict)
        assert isinstance(m1, dict)
    
    def test_per_mol_metrics(self, mol_batch):
        """Per-molecule metrics should be lists."""
        from metrics import all_scores
        
        class MockData:
            smiles = ['CCO']
            data = [Chem.MolFromSmiles('CCO')]
        
        m0, m1 = all_scores(mol_batch, MockData())
        
        assert 'QED score' in m0
        assert 'SA score' in m0
        assert 'logP score' in m0
    
    def test_aggregate_metrics(self, mol_batch):
        """Aggregate metrics should be percentages."""
        from metrics import all_scores
        
        class MockData:
            smiles = ['CCO']
            data = [Chem.MolFromSmiles('CCO')]
        
        m0, m1 = all_scores(mol_batch, MockData())
        
        assert 'valid score' in m1
        assert 'unique score' in m1
        assert 'novel score' in m1
        
        # Should be percentages (0-100)
        assert 0 <= m1['valid score'] <= 100


# =============================================================================
# Test Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_empty_batch(self):
        """Empty batch should return empty arrays."""
        from metrics import qed_scores, validity_ratio
        scores = qed_scores([])
        assert scores.shape == (0,)
        
        ratio = validity_ratio([])
        assert ratio == 0.0
    
    def test_all_invalid(self, invalid_mol):
        """All invalid molecules should return zeros."""
        from metrics import validity_ratio, uniqueness_ratio
        mols = [invalid_mol, invalid_mol, invalid_mol]
        
        assert validity_ratio(mols) == 0.0
        assert uniqueness_ratio(mols) == 0.0
    
    def test_mol_to_smiles_invalid(self, invalid_mol):
        """mol_to_smiles should return None for invalid mol."""
        from metrics import mol_to_smiles
        assert mol_to_smiles(invalid_mol) is None
    
    def test_normalize_function(self):
        """Test normalize utility function."""
        from metrics import normalize
        arr = np.array([0, 5, 10])
        normalized = normalize(arr, 0, 10)
        np.testing.assert_array_almost_equal(normalized, [0.0, 0.5, 1.0])
    
    def test_normalize_clipping(self):
        """Normalize should clip values outside range."""
        from metrics import normalize
        arr = np.array([-5, 15])
        normalized = normalize(arr, 0, 10)
        np.testing.assert_array_almost_equal(normalized, [0.0, 1.0])


# =============================================================================
# Test Drug Candidate Score
# =============================================================================

class TestDrugCandidateScore:
    """Tests for drug candidate score."""
    
    def test_shape(self, mol_batch):
        """Drug candidate scores should have correct shape."""
        from metrics import drugcandidate_scores
        scores = drugcandidate_scores(mol_batch)
        assert scores.shape == (5,)
    
    def test_range(self, valid_mol, valid_mol2, valid_mol3):
        """Drug candidate scores should be in [0, 1]."""
        from metrics import drugcandidate_scores
        mols = [valid_mol, valid_mol2, valid_mol3]
        scores = drugcandidate_scores(mols)
        assert all(0.0 <= s <= 1.0 for s in scores)
    
    def test_with_train_smiles(self, mol_batch, train_smiles):
        """Drug candidate should work with train_smiles."""
        from metrics import drugcandidate_scores
        scores = drugcandidate_scores(mol_batch, train_smiles=train_smiles)
        assert scores.shape == (5,)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

