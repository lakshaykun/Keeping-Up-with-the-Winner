import io
import numpy as np
import networkx as nx
import pandas as pd


def load_graph_from_edgelist(path_or_bytes) -> nx.Graph:
    """Load an undirected graph from an edgelist file (u v per line).
    Accepts a file path or an in-memory bytes object (for Streamlit uploads).
    """
    if isinstance(path_or_bytes, (bytes, bytearray)):
        f = io.StringIO(path_or_bytes.decode("utf-8"))
        G = nx.read_edgelist(f, nodetype=int)
    else:
        G = nx.read_edgelist(path_or_bytes, nodetype=int)
    if not isinstance(G, nx.Graph):
        G = nx.Graph(G)
    # Ensure connected component (largest) like paper assumes connected graphs
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
    return G


def load_graph_from_adj_csv(path_or_bytes) -> nx.Graph:
    """Load adjacency matrix CSV (square numeric) into an undirected graph."""
    if isinstance(path_or_bytes, (bytes, bytearray)):
        df = pd.read_csv(io.BytesIO(path_or_bytes), header=None)
    else:
        df = pd.read_csv(path_or_bytes, header=None)
    A = df.values
    A = np.where(A > 0, 1, 0)
    G = nx.from_numpy_array(A)
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
    return G


def graph_to_adjacency(G: nx.Graph) -> np.ndarray:
    return nx.to_numpy_array(G, dtype=float)


def ensure_symmetric_binary(A: np.ndarray) -> np.ndarray:
    A = (A + A.T) / 2.0
    A = (A > 0).astype(float)
    np.fill_diagonal(A, 0.0)
    return A


def normalize_prob_vector(v: np.ndarray) -> np.ndarray:
    v = np.maximum(v, 0)
    s = v.sum()
    if s <= 0:
        return np.zeros_like(v)
    return v / s