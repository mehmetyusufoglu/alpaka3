# Collective FFT Data Flow

This table tracks where the distributed FFT demo stores intermediate state, how large each buffer is, and when the phase correction happens. Assume there are 4 GPUs.

| Stage | Data Description | Location | Approximate Size | Notes |
| - | - | - | - | - |
| Load strided samples | `localSamples` (rank’s slice of original signal) | Host | `N/4` complex floats | Reads from file or synthetic generator before GPU staging. |
| Upload to GPU | `deviceInput` | GPU | `N/4` complex floats | Copies the rank’s slice into the device tensor managed by `CleanTensorOpContext`. |
| FFT execution | `deviceInput` → `deviceOutput` | GPU | `N/4` complex floats | `context.fft` triggers cuFFT (or fallback) to produce the local spectrum. |
| Download FFT output | `deviceOutput` → `localSpectrum` | Host | `N/4` complex floats | Pulls the spectrum back to CPU so the next steps can run without extra device memory. |
| Phase correction | `detail::buildContributionTile` input | Host | `N/4` complex floats | Applies lag-dependent phase factors per global bin; occurs entirely on CPU. |
| Tile staging | `tileContributions` slice → `spectralTensor` | Host → GPU | `tileCapacity` complex floats (default 262 144) | Packs the current slice’s real/imag pairs and uploads only that tile to the device. |
| NCCL all-reduce | `spectralTensor` | GPU | `tileCapacity` complex floats | Runs tiled all-reduce so ranks sum matching bins without allocating the full `N` spectrum on device. |
| Tile download | `spectralTensor` → host buffer (`globalSpectrum` / preview) | Host | `tileCapacity` complex floats | Copies the reduced tile back and appends it to the global spectrum or preview buffer. |
| Final aggregation | `globalSpectrum` (optional) | Host | `N` complex floats | Retains the complete spectrum if verification or reference comparison is requested. |
