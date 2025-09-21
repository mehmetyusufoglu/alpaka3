# Alpaka-AI Architecture Docs

Companion documentation set focused on the Alpaka Tensor Library (AI-centric). Mirrors structure of `alpaka-architecture-docs` and extends it with tensor-specific analysis, diagrams, and examples.

## Structure
- analysis/ — deep-dives, design notes, API walkthroughs
- config/ — PlantUML/diagram configs
- diagrams/ — .puml sources
- output/ — generated artifacts (PNGs/SVGs)
- scripts/ — helper scripts for generation

## Getting Started
- Add content to `analysis/` as markdown
- Put PlantUML sources in `diagrams/` and run scripts in `scripts/` to generate images into `output/`

## Scope
- Tensor core types, memory/layout, ops (conv2d, pooling, normalization, activation)
- Execution model and providers (CPU, CUDA/HIP, SYCL)
- Interop with Alpaka core (queues, executors, views)
