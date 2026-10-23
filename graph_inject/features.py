"""Structural node features computed once per graph (CPU, numpy/PyG).

  RWSE   random-walk structural encoding: diag((D^-1 A)^t), t = 1..walk_length (PyG AddRandomWalkPE).
  LapPE  first k non-trivial eigenvectors of the normalized Laplacian (PyG AddLaplacianEigenvectorPE).
         Eigenvectors are defined up to sign: methods flip signs at random during training.
  hops   LLaGA hop-field features: x, A_hat x, A_hat^2 x, ... with A_hat = D^-1/2 A D^-1/2
         (VITA-Group/LLaGA utils/data_process.py generate_multi_hop_x, no self loops).
"""
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import AddLaplacianEigenvectorPE, AddRandomWalkPE

from graph_inject.structure import PromptGraph


def _pyg_data(graph: PromptGraph) -> Data:
    if graph.edges:
        edges = torch.tensor(graph.edges, dtype=torch.long).T
        edge_index = torch.cat([edges, edges.flip(0)], dim=1)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    return Data(edge_index=edge_index, num_nodes=graph.nnodes)


def random_walk_pe(graph: PromptGraph, walk_length: int) -> np.ndarray:
    data = AddRandomWalkPE(walk_length=walk_length, attr_name="pe")(_pyg_data(graph))
    return data.pe.numpy().astype(np.float32)


def laplacian_pe(graph: PromptGraph, k: int) -> np.ndarray:
    """[nnodes, k]; graphs with fewer than k + 1 nodes are zero-padded."""
    out = np.zeros((graph.nnodes, k), dtype=np.float32)
    k_eff = min(k, graph.nnodes - 1)
    if k_eff > 0:
        transform = AddLaplacianEigenvectorPE(k=k_eff, attr_name="pe", is_undirected=True)
        # PyG's dense eigh at every size: its sparse ARPACK path (>= 100 nodes, KG prompts) starts from a random
        # vector and fails to converge on some disconnected graphs. Below 100 nodes this is PyG's default path.
        transform.SPARSE_THRESHOLD = 1 << 30
        data = transform(_pyg_data(graph))
        out[:, :k_eff] = data.pe.numpy()
    return out


def structural_features(graph: PromptGraph, walk_length: int = 16, lap_k: int = 8) -> np.ndarray:
    """[nnodes, walk_length + lap_k]: RWSE followed by LapPE (the LapPE block is sign-flipped in training)."""
    if graph.nnodes == 0:  # KQA Pro prompts without facts
        return np.zeros((0, walk_length + lap_k), dtype=np.float32)
    return np.concatenate([random_walk_pe(graph, walk_length), laplacian_pe(graph, lap_k)], axis=1)


def normalized_adjacency(graph: PromptGraph) -> np.ndarray:
    adj = np.zeros((graph.nnodes, graph.nnodes), dtype=np.float32)
    for u, v in graph.edges:
        adj[u, v] = adj[v, u] = 1.0
    deg = adj.sum(1)
    inv_sqrt = np.where(deg > 0, deg ** -0.5, 0.0).astype(np.float32)
    return inv_sqrt[:, None] * adj * inv_sqrt[None, :]


def hop_features(x: torch.Tensor, a_hat: torch.Tensor, hops: int) -> torch.Tensor:
    """[nnodes, hops + 1, F]: LLaGA hop-field template, computed on the fly (x may be trainable)."""
    out = [x]
    for _ in range(hops):
        out.append(a_hat @ out[-1])
    return torch.stack(out, dim=1)
