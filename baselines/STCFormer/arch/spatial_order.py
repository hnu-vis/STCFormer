"""Static, coordinate-only alternatives to serpentine patch construction."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def hilbert_order(coordinates, bits=16):
    # Local equirectangular projection, common scale (no aspect-ratio stretching).
    ll = np.deg2rad(np.asarray(coordinates, dtype=np.float64)[:, :2])
    xy = np.column_stack((ll[:, 0] * np.cos(ll[:, 1].mean()), ll[:, 1]))
    xy -= xy.min(axis=0)
    grid = np.rint(xy / max(xy.max(), 1e-12) * (2**bits - 1)).astype(np.int64)
    x, y = grid.T.copy()
    distance = np.zeros(len(x), dtype=np.int64)
    s = 2**(bits - 1)
    while s:
        rx, ry = (x & s) > 0, (y & s) > 0
        distance += s*s * ((3*rx.astype(np.int64)) ^ ry.astype(np.int64))
        flip = (~ry) & rx
        x[flip], y[flip] = s-1-x[flip], s-1-y[flip]
        swap = ~ry
        x[swap], y[swap] = y[swap].copy(), x[swap].copy()
        s //= 2
    return np.argsort(distance, kind='stable')


def spatial_order(coordinates, method, patch_size, seed=2024):
    coords = np.asarray(coordinates)
    if not np.isfinite(coords).all() or len(coords) == 0 or patch_size < 1:
        raise ValueError('Finite nonempty coordinates and positive patch size required')
    if method == 'random':
        return np.random.default_rng(seed).permutation(len(coords))
    initial = hilbert_order(coords)
    if method == 'hilbert':
        return initial
    if method != 'capacity':
        raise ValueError(method)
    # Equal-capacity Lloyd clustering in unit-sphere chord distance. Each center
    # is expanded into capacity-many slots; Hungarian assignment enforces exact
    # capacities. Last group contains N % patch_size actual stations.
    lon, lat = np.deg2rad(coords[:, :2]).T
    xyz = np.column_stack((np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)))
    groups = [initial[i:i+patch_size] for i in range(0, len(coords), patch_size)]
    capacities = np.array([len(g) for g in groups])
    centers = np.array([xyz[g].mean(axis=0) for g in groups])
    slots = np.repeat(np.arange(len(groups)), capacities)
    previous = None
    for _ in range(20):
        rows, columns = linear_sum_assignment(cdist(xyz, centers, 'sqeuclidean')[:, slots])
        labels = np.empty(len(coords), dtype=np.int64)
        labels[rows] = slots[columns]
        if previous is not None and np.array_equal(labels, previous):
            break
        centers = np.array([xyz[labels == k].mean(axis=0) for k in range(len(groups))])
        previous = labels
    # Canonical intra-group order; last (possibly incomplete) group stays last.
    rank = np.empty(len(coords), dtype=np.int64)
    rank[initial] = np.arange(len(coords))
    return np.lexsort((rank, labels))
