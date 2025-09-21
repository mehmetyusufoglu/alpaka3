# Alpaka Architecture Documentation

## Overview

This project provides a comprehensive documentation of the Alpaka portability library, focusing on its architecture, components, and the relationships between them. The documentation includes diagrams, analysis of the directory structure, and scripts for automating tasks related to the documentation process.

## Directory Structure

The project is organized into several directories, each serving a specific purpose:

- **diagrams/**: Contains PlantUML files that illustrate various aspects of the Alpaka architecture.
  - `high-level-architecture.puml`: High-level block diagram of the Alpaka architecture.
  - `abstraction-hierarchy.puml`: Detailed UML diagram showing the hierarchy of abstractions within the Alpaka library.
  - `memory-model.puml`: Depicts the memory model used in Alpaka.
  - `parallelism-levels.puml`: Illustrates the different levels of parallelism supported by Alpaka.

- **analysis/**: Contains markdown files that document the analysis of the Alpaka library.
  - `directory-structure.md`: Documentation of the structure of the Alpaka directory.
  - `component-analysis.md`: Analysis of the components within the Alpaka library.
  - `dependency-graph.md`: Dependency graph showing relationships between components.
  - `memory-system.md`: Comprehensive analysis of Alpaka's memory allocation system, View/MdSpan abstractions, and shared memory usage.
  - `namespace-organization.md`: Complete guide to Alpaka's namespace hierarchy and organization patterns.

- **scripts/**: Contains scripts for analyzing header files and generating diagrams.
  - `analyze-headers.py`: Script to analyze header files in the Alpaka library.
  - `generate-diagrams.sh`: Script to automate the generation of diagrams.

- **output/**: Directory to store output files, kept in version control with a `.gitkeep` file.

- **config/**: Contains configuration settings for PlantUML.
  - `plantuml-config.iuml`: Configuration file for customizing generated diagrams.

- **.gitignore**: Specifies files and directories to be ignored by Git.

## Key Architectural Concepts

### Three-Layer Abstraction Model

Alpaka's innovative design separates concerns through three distinct layers:

1. **APIs (Programming Models)** - Define how parallel code is written:
   - `api::host`: Standard C++ with threading libraries
   - `api::cuda`: NVIDIA CUDA programming model
   - `api::hip`: AMD HIP programming model
   - `api::oneApi`: Intel OneAPI/SYCL standard

2. **Device Kinds (Hardware Categories)** - Classify computational hardware:
   - `deviceKind::cpu`: Multi-core processors
   - `deviceKind::amdGpu`: AMD graphics processors
   - `deviceKind::nvidiaGpu`: NVIDIA graphics processors
   - `deviceKind::intelGpu`: Intel graphics processors

3. **Executors (Execution Strategies)** - Combine APIs with parallelization approaches:
   - `exec::cpuSerial`: Sequential CPU execution
   - `exec::ompBlocks`: OpenMP block-parallel execution
   - `exec::gpuCuda`: CUDA GPU execution strategy
   - `exec::gpuHip`: HIP GPU execution strategy
   - `exec::oneApi`: OneAPI execution (CPU or GPU)

This model enables flexible hardware mapping and future extensibility.

### Compile-Time vs Runtime Selection

Alpaka employs a sophisticated selection model separating:

- **Compile-Time**: Template instantiation for all backends, zero-overhead dispatch
- **Runtime**: Hardware detection, device selection, and resource creation

Key components:
- **Backend Dictionaries**: `Dict{DeviceSpec, Executor}` combinations
- **makeAcc Function**: Creates accelerator contexts for kernel execution  
- **Template Instantiation**: Separate kernel compilation per backend

This enables both performance (compile-time optimization) and flexibility (runtime hardware discovery).

## Analysis Plan

The analysis of the Alpaka directory structure will follow these steps:

1. **Analyze the `include/alpaka` Directory**: Review header files, noting classes, functions, and key components.
2. **Analyze the `core` Subdirectory**: Document the functionality and purpose of core components.
3. **Analyze the `api` Subdirectory**: Review API components and their interactions.
4. **Analyze the `onAcc` and `onOcc` Subdirectories**: Examine components specific to accelerator and offloading contexts.
5. **Compile Findings**: Summarize architecture and hierarchy of classes, and create diagrams.
6. **Review and Refine**: Ensure accuracy of diagrams and documents, making adjustments as necessary.

## Usage

To generate diagrams and analyze the Alpaka library, run the provided scripts in the `scripts/` directory. Ensure that the necessary dependencies for PlantUML are installed and configured according to the settings in `config/plantuml-config.iuml`.

## Conclusion

This documentation aims to provide a clear understanding of the Alpaka library's architecture and its components, facilitating easier usage and further development.