# tensorFft3D

`tensorFft3D` demonstrates how to execute batched 3-D complex FFTs through the
Alpaka tensor provider interface. When CUDA and cuFFT are available the example
runs the transforms on the GPU via the new `CuFFTProvider`. Otherwise it falls
back to the generic host implementation.

## Build

```bash
cmake --build build --target tensorFft3D -j8
```

## Usage

```bash
./example/tensor/fft3D/tensorFft3D [options]
```

Options:

- `--size=<N>`: Set identical dimensions for x/y/z (default 16).
- `--nx=<N>`, `--ny=<N>`, `--nz=<N>`: Override individual extents.
- `--batch=<B>`: Number of independent transforms per launch (default 1).
- `--no-verify`: Skip the inverse-transform accuracy check.
- `--verbose`: Print the first few frequency-domain samples.
- `--force-provider-fft`: Require cuFFT; the program exits with an error if the
  provider is unavailable.

The example reports the maximum absolute and relative error after performing a
forward + inverse transform pair. When cuFFT is not present the fallback path
executes a naive O(N²) transform on the host, which is only practical for the
default (16³) problem size.
