from itertools import combinations
import numpy as np

def filter(x, d=0.1):
    # lim0 = 2*d/np.sqrt(np.pi)
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    mask0 = x == 0
    xnon0 = x[~mask0]
    out[mask0] = 1
    if np.any(~mask0):
        out[~mask0] = np.vectorize(np.math.erf)(xnon0/d)/(xnon0)
    return out

def get_mo_degeneracy(mo_energies, d=0.1):
    M = mo_energies.shape[1]
    cumulative = []
    for r in range(2, M):
        _cumulative = []
        for comb in combinations(range(M), r=r):
            compare = comb[0]
            select  = np.array(comb[1:])
            diff = mo_energies[:, compare] - np.sum(mo_energies[:,select], axis=1)
            _cumulative.append(diff)
        _cumulative = np.sum(filter(_cumulative, d=d),axis=0) / np.math.comb(M, r)
        cumulative.append(_cumulative)
    cumulative = d * np.sqrt(np.pi) / 2 * np.mean(cumulative, axis=0) # / (2**(M-1))
    return cumulative