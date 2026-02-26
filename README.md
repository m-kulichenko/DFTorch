# DFTorch

DFTorch is a Density Functional Tight Binding (DFTB) implementation in PyTorch.

## Installation

### uv (recommended)

```bash
cd DFTorch
uv venv --python 3.11
uv pip install .
```

Run tests:
```bash
uv run pytest
```

### pip
To install DFTorch, run:

```bash
pip install .
```


## Requirements
- torch
- numpy
- scipy
- pandas

## Usage

See `experiments/1_tutorial.ipynb` for examples.

### Public API

Supported public imports:

```python
from dftorch import (
    Constants,
    Structure,
    StructureBatch,
    ESDriver,
    ESDriverBatch,
    MDXL,
    MDXLBatch,
    MDXLOS,
)
```

All other modules are internal implementation details and may change.

This folder contains legacy code kept for reference.

## Authors
 M. Kulichenko, A.M.N. Niklasson
