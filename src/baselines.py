import numpy as np
import networkx as nx
from numpy.linalg import eig


def baseline_degree(A):
    G = nx.from_numpy_array(A)
    deg = np.array([d for _, d in G.degree()], dtype=float)
    if deg.sum() == 0:
        return np.zeros(A.shape[0])
    return deg / deg.sum()


def baseline_eigenvector(A):
    G = nx.from_numpy_array(A)
    ev = nx.eigenvector_centrality_numpy(G)
    v = np.array([ev[i] for i in range(A.shape[0])], dtype=float)
    if v.sum() == 0:
        return np.zeros_like(v)
    return v / v.sum()


def baseline_netshield(A, k):
    # Simplified NetShield-like selection: pick top-k by squared PF eigenvector
    w, V = eig(A)
    idx = np.argmax(w.real)
    u = V[:, idx].real
    score = u ** 2
    S = np.argsort(-score)[:k]
    sel = np.zeros(A.shape[0])
    sel[S] = 1.0
    return sel