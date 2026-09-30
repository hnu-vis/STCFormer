"""Dataset-specific, deterministic METIS and position features (no target data)."""

import hashlib
import os
import pickle
from pathlib import Path

import numpy as np
import pymetis
from scipy.linalg import eigh
from scipy.special import sph_harm
from threadpoolctl import threadpool_limits


def prepare_graph(root_path, num_nodes, num_parts=32, node_dim=48):
    root = Path(root_path)
    adj_path = root / "adj_mx.pkl"
    coord_path = root / "stations_sorted_reduced.npy"
    digest = hashlib.sha256(adj_path.read_bytes() + coord_path.read_bytes()).hexdigest()[:16]
    parts = min(num_parts, num_nodes)
    cache = Path(__file__).resolve().parents[3] / "artifacts" / "cache" / "s2transformer"
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"{root.name}_{digest}_p{parts}_d{node_dim}_v1.npz"
    if path.exists():
        with np.load(path) as data:
            return {key: data[key] for key in data.files}
    with adj_path.open("rb") as file:
        adjacency = pickle.load(file)
    if isinstance(adjacency, (tuple, list)):
        adjacency = adjacency[-1]
    adjacency = np.asarray(adjacency, dtype=np.float64)
    if adjacency.shape != (num_nodes, num_nodes) or not np.isfinite(adjacency).all():
        raise ValueError("Invalid station adjacency")
    adjacency = np.maximum(adjacency, adjacency.T)
    np.fill_diagonal(adjacency, 0)
    neighbors = [np.flatnonzero(row > 0).tolist() for row in adjacency]
    # Preserve upstream weighted recursive METIS and its fixed geography seed.
    xadj = np.r_[0, np.cumsum([len(row) for row in neighbors])].tolist()
    adjncy = [node for row in neighbors for node in row]
    weights = [max(1, int(adjacency[i, j] * 100))
               for i, row in enumerate(neighbors) for j in row]
    _, membership = pymetis.part_graph(
        parts, xadj=xadj, adjncy=adjncy, eweights=weights,
        recursive=True, options=pymetis.Options(seed=2025),
    )
    groups = [np.flatnonzero(np.asarray(membership) == i) for i in range(parts)]
    if any(len(group) == 0 for group in groups):
        raise ValueError("METIS returned an empty partition")
    patch = np.full((parts, max(map(len, groups))), num_nodes, dtype=np.int64)
    intra = np.zeros((parts, patch.shape[1], patch.shape[1]), dtype=np.float32)
    inter = np.eye(parts, dtype=np.float32)
    for i, group in enumerate(groups):
        patch[i, :len(group)] = group
        intra[i, :len(group), :len(group)] = adjacency[np.ix_(group, group)]
        np.fill_diagonal(intra[i], 1)
        for j, other in enumerate(groups):
            if i != j:
                inter[i, j] = adjacency[np.ix_(group, other)].max()
    degree = adjacency.sum(1)
    inv = 1 / np.sqrt(np.maximum(degree, 1e-12))
    laplacian = np.eye(num_nodes) - inv[:, None] * adjacency * inv[None, :]
    with threadpool_limits(limits=4):
        _, vectors = eigh(laplacian, subset_by_index=[0, min(node_dim, num_nodes)-1])
    # Fix otherwise arbitrary eigenvector signs for reproducible preprocessing.
    signs = np.sign(vectors[np.abs(vectors).argmax(0), np.arange(vectors.shape[1])])
    vectors *= np.where(signs == 0, 1, signs)
    vectors = np.pad(vectors, ((0, 0), (0, node_dim-vectors.shape[1])))
    coords = np.load(coord_path)
    # Global station metadata is [latitude, longitude, altitude]; the regional
    # datasets are [longitude, latitude, altitude]. Never infer from row order.
    lon, lat = (coords[:, 1], coords[:, 0]) if root.name.startswith("Global") else (coords[:, 0], coords[:, 1])
    if np.max(np.abs(lat)) > 90 or np.max(np.abs(lon)) > 180:
        raise ValueError("Invalid latitude/longitude column mapping")
    azimuth, polar = np.deg2rad(lon), np.deg2rad(90-lat)
    harmonics = []
    for ell in range(4):
        for m in range(-ell, ell+1):
            y = sph_harm(abs(m), ell, azimuth, polar)
            harmonics.append(y.real if m == 0 else
                             np.sqrt(2)*(-1)**m*(y.imag if m < 0 else y.real))
    result = dict(patch=patch, intra=intra, inter=inter,
                  node_features=vectors.astype(np.float32),
                  spherical=np.stack(harmonics, -1).astype(np.float32))
    temporary = path.with_suffix(f".tmp.{os.getpid()}.npz")
    np.savez(temporary, **result)
    temporary.replace(path)
    return result
