# CDPNet source note

- Official repository: <https://github.com/ChujieXu/CDPNet>, inspected from
  its 2025 MIT-licensed release.
- The upstream implementation assumes batch size one, hard-codes CUDA device
  construction and a Weather2K-specific directory layout.  This directory is
  a clean fixed-station BasicTS adaptation using the repository's two defining
  paths: calibrated station-to-grid initialization and recurrent diffusive
  difference estimation followed by grid-to-station interpolation.
- Station coordinates are loaded from each BasicTS dataset's
  `stations_sorted_reduced.npy`; no extra runtime dependency is required.
