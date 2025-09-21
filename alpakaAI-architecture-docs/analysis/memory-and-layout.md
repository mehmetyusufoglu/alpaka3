# Memory and Layout — Alpaka Tensor

## Ownership & Coherence
- Tensors own host and (lazily) device buffers
- CoherenceState tracks which side is freshest: HostFresh, DeviceFresh, BothFresh
- `ensureOnDevice(device, queue)` copies host→device if needed
- `toHost(device, queue)` copies device→host when DeviceFresh

## Layout
- Current implementation assumes row-major contiguous layout (NCHW for 4D)
- `TensorDescriptor` used for asserts in debug builds
- Future: add explicit layout tags and strides

## Access
- `hostData()` mutable pointer; call `markHostModified()` after edits
- `deviceBuffer(device, queue)` returns backend buffer (allocates lazily)

## Launch Shapes
- Ops use `onHost::FrameSpec{numFrames, frameExtent}`
- Kernels iterate with `onAcc::makeIdxMap(acc, worker::threadsInGrid, IdxRange{...})`

## Synchronization
- Many ops avoid immediate `wait()`; prefer lazy synchronization
- Host copies (`toHost`) imply synchronization
