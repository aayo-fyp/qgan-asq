from data.sparse_molecular_dataset import SparseMolecularDataset
from utils import MolecularMetrics
import numpy as np, random

print('Loading qm9 dataset...')
d = SparseMolecularDataset()
d.load('data/gdb9.sparsedataset')
print('dataset size:', len(d.data))

mols = list(d.data)
# sample up to 2000 molecules
n = min(2000, len(mols))
sample = random.sample(mols, n)
print('sampling', n, 'molecules')

np_scores = MolecularMetrics.natural_product_scores(sample, norm=True)
sa_scores = MolecularMetrics.synthetic_accessibility_score_scores(sample, norm=True)

print('NP scores: mean={:.4f}, median={:.4f}, zeros={}/{} ({:.2f}%)'.format(np.nanmean(np_scores), np.median(np_scores), np.sum(np_scores==0), n, 100*np.sum(np_scores==0)/n))
print('SA scores: mean={:.4f}, median={:.4f}, zeros={}/{} ({:.2f}%)'.format(np.nanmean(sa_scores), np.median(sa_scores), np.sum(sa_scores==0), n, 100*np.sum(sa_scores==0)/n))

# show small histogram bins
bins = np.linspace(0,1,11)
np_hist = np.histogram(np_scores, bins=bins)[0]
sa_hist = np.histogram(sa_scores, bins=bins)[0]
print('\nNP histogram (0.0-1.0, 10 bins):', np_hist)
print('SA histogram (0.0-1.0, 10 bins):', sa_hist)
