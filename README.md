# DFTorch

DFTorch is a Density Functional Tight Binding (DFTB) implementation in PyTorch.

## Installation

### uv (recommended)

```bash
cd DFTorch
uv venv --python 3.11
uv pip install -e ".[dev]"
```


### pip
To install DFTorch, run:

```bash
pip install .
```
Run tests:
```bash
uv run pytest
```

## Requirements
- torch
- numpy
- scipy
- pandas

## Usage

Import modules from DFTorch after installation:

see /experiments/1_tutorial.ipynb for examples.

## Authors
 M. Kulichenko, A.M.N. Niklasson
