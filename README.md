# LLM-CiM

Software infrastructure for evaluating an analog Compute-in-Memory (CiM) co-processor for low-latency edge LLM inference.

## Overview

This repository contains the code and experiment artifacts used to evaluate a hybrid CPU–CiM architecture for transformer inference. The proposed system selectively offloads weight-stationary GEMV operations from projection and feed-forward layers to a CiM co-processor, while keeping attention, nonlinear operations, and control flow on the CPU.

The repository includes:

- MLIR generation scripts for GPT and Phi models.
- Scripts used to generate GEMV and non-GEMV model variants.
- A gem5 configuration script for full-system simulation.
- Version-pinned simulation backends through Git submodules:
  - gem5 for architectural simulation.
  - MNSIM for CiM modeling and result generation.

The result analysis and visualization flow is maintained inside the MNSIM submodule, where the CiM simulation outputs are post-processed into plots and CSV files.

## Repository Structure

```text
llm-cim/
├── gem5/              # gem5 submodule
├── MNSIM/             # MNSIM submodule
├── gpt/               # GPT model scripts and workflow notes
├── phi/               # Phi model scripts and workflow notes
├── .gitmodules
└── README.md
```
## Reproducibility

Clone the repository together with all pinned dependencies:

```bash
git clone --recursive https://github.com/Aishneet/llm-cim.git
```

or initialize submodules after cloning:

```bash
git submodule update --init --recursive
```

The exact simulator versions used in the experiments are pinned through the gem5 and MNSIM submodules.

## Workflow

1. Generate MLIR from GPT or Phi models.
2. Lower MLIR to LLVM IR and compile to x86 binaries.
3. Execute CPU-only workloads using gem5.
4. Model CiM-accelerated GEMV execution using MNSIM.

Detailed instructions are available in:

- `gpt/README.md`
- `phi/README.md`
