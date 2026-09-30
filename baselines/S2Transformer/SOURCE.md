# S²Transformer — BasicTS weather adaptation

Source: https://github.com/hongyichenhitsz/S2Transformer
Pinned commit: `2f028c2247fffdade87fa3ec993461d82a4fcb5a`.
Paper: https://openreview.net/pdf?id=AL2VnKno5n (TMLR 2026).
Original downloaded model, attention, graph utilities, experiment builder and
README are retained unmodified in `upstream/` for provenance. The pinned upstream
tree has no license file; do not assume a permissive redistribution license.

## Preserved released architecture

History-to-station projection, concatenated spectral/calendar/spherical features,
recursive weighted METIS with seed 2025, intra-partition spatial attention,
pooled inter-partition attention, residual FFNs, concatenation fusion, and direct
multi-horizon projection. Like the released `exp_main.py`, both encoder blocks
reuse the same 32 partitions and use weighted adjacency as attention bias.
This is the **released-code variant**, not a claim to implement the paper's
multiscale/shortest-path variant when it differs from the public release.

## Necessary adaptations and controlled benchmark settings

- Dynamic 234/3570/3850 station counts, registered geographic buffers, no hardcoded
  external data paths. Input `[B,48,N,5]`, output `[B,H,N,1]`; no future labels used.
- Regenerate METIS, normalized-Laplacian eigenvectors (48 dimensions), and real
  degree-0..3 spherical harmonics (16 dimensions) from each dataset's own graph
  and coordinates. Global coordinate columns are latitude/longitude, while
  French/Hunan columns are longitude/latitude. Cache keys include graph/coord hash.
- Existing prepared calendar channels are hour/24, weekday/7,
  (day-of-year-1)/366, (month-1)/12, despite stale dataset description labels.
  Use these four available channels, with 16-dimensional embeddings each,
  instead of the upstream month/day-of-month/weekday/hour fields (32 each).
- Total attention width 256 = value 128 + spectral 48 + calendar 64 + spherical 16;
  FFN width 512, 8 heads, 2 blocks, dropout 0.1. Upstream default total width is 960.
- PyTorch SDPA retains both learnable geographic bias and padding mask. It does
  not use the upstream fused branch that silently discards the bias. Padding
  mean pooling is retained; repeated dummy-node scatter writes are eliminated.
- Same BasicTS splits/scaling/masked-MAE objective, AdamW, LR 3e-4, best-validation
  MAE selection, and 30 epochs as this repository's main baseline protocol.
  This intentionally differs from upstream MSE/Adam; do not label these runs an
  exact reproduction of published scores.
- Horizons 24/72, history 48; seeds 2024/2025, 8 datasets = 32 jobs.
- MAE/MSE/SEDI and mean/population variance/std use the existing result collector.
  Global raw-unit metrics remain raw in JSON (reporting scales MAE /10, MSE /100).

## Environment and execution

Use `/mnt/HNU/conda_envs/BasicTS/bin/python`. Additional dependency:
`pymetis==2023.1.1`; existing NumPy/SciPy/PyTorch suffice (no timm required).

```bash
/mnt/HNU/conda_envs/BasicTS/bin/python scripts/run_weather_seeded_experiments.py \
  --models S2Transformer --gpus 3 4 5 6 7 --slots-per-gpu 1 \
  --horizons 24 72 --seeds 2024 2025 --status-name status_s2transformer.json
```

Existing results are skipped; OOM retries halve batch size down to 1. Results:
`artifacts/results/weather_48_24_72_30e/S2Transformer/`.

Validation: `scripts/check_s2transformer.py` checks real train-prefix windows,
both horizons, finite outputs/gradients, learned bias gradients, SDPA versus
explicit attention, partition coverage and evaluation independence of future
labels. All 8 datasets passed. French batch 512 peaks at 6.35 GiB; Global batch
112 peaks at 17.54 GiB in model/optimizer diagnostics (runner overhead is extra).
These are the initial French/large batch sizes, with OOM halving enabled.
