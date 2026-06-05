# MLIR Compilation Workflow

This document describes the common workflow used to compile generated MLIR models into executable x86 binaries. The same flow is used for both GPT and Phi model families.

## Overview

The compilation pipeline consists of:

1. Generate linalg-based MLIR from a model script.
2. Lower MLIR to LLVM-compatible MLIR.
3. Translate MLIR to LLVM IR.
4. Generate an object file.
5. Rename the generated `main` symbol.
6. Compile a runtime driver.
7. Link the final executable.

## LLVM Setup

Set the LLVM build directory:

```bash
export LLVM_BUILD=<path-to-llvm-build>
```

---

## Step 1: Lower MLIR

```bash
$LLVM_BUILD/bin/mlir-opt model.mlir \
  --one-shot-bufferize="allow-unknown-ops function-boundary-type-conversion=identity-layout-map bufferize-function-boundaries=1" \
  --convert-linalg-to-loops \
  --expand-strided-metadata \
  --lower-affine \
  --convert-scf-to-cf \
  --convert-math-to-llvm \
  --convert-cf-to-llvm \
  --finalize-memref-to-llvm \
  --convert-arith-to-llvm \
  --convert-func-to-llvm \
  --reconcile-unrealized-casts \
  -o model_lowered.mlir
```

---

## Step 2: Convert to LLVM IR

```bash
$LLVM_BUILD/bin/mlir-translate \
  --mlir-to-llvmir \
  model_lowered.mlir \
  -o model.ll
```

---

## Step 3: Generate Object File

```bash
$LLVM_BUILD/bin/llc \
  -mtriple=x86_64-pc-linux-gnu \
  -mcpu=generic \
  -filetype=obj \
  -O2 \
  model.ll \
  -o model.o
```

---

## Step 4: Rename the Entry Point

```bash
llvm-objcopy \
  --redefine-sym main=model_main \
  model.o \
  model_model.o
```

### Why rename `main`?

The MLIR-generated module contains its own `main()` function. The runtime driver also contains a `main()` function. Without renaming, linking fails due to multiple definitions of `main`. The generated entry point is renamed to:

```c
model_main(...)
```

which is then called by the runtime driver.

---

## Step 5: Compile the Runtime Driver

```bash
clang -O2 -c m.c -o mG.o
```

<!-- For large models (e.g., GPT-2 XL):

```bash
clang -O2 -mcmodel=large -c m.c -o mG.o
```
-->
---

## Step 6: Link the Final Binary

### GPT Models

```bash
clang -no-pie mG.o model_model.o -lm -o model_x86
```

### Phi Models

```bash
clang \
  -no-pie \
  mG.o \
  model_model.o \
  -L$LLVM_BUILD/lib \
  -lmlir_c_runner_utils \
  -lm \
  -o model_x86
```

<!--For larger models:

```bash
clang \
  -no-pie \
  -mcmodel=large \
  mG.o \
  model_model.o \
  -lm \
  -o model_x86
```
-->
---

## Step 7: Execute

### GPT

```bash
./model_x86
```

### Phi

```bash
export LD_LIBRARY_PATH=$LLVM_BUILD/lib:$LD_LIBRARY_PATH

./model_x86
```

---

## Outputs

The generated executable binaries are executed in gem5 to obtain CPU latency measurements. The resulting measurements are used to:

- Evaluate the CPU-only baseline.
- Measure the CPU-side execution time of the hybrid CPU–CiM architecture.
- Compute end-to-end hybrid CPU–CiM latency when combined with CiM latency estimates from MNSIM