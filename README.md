# Crystallographic graph

Generate knn graph and cutoff graph from crystallographic materials with a GPU support

## Usage

### K nearest neighbors graph

```python
from crystallographic_graph import make_graph

edges = make_graph(cell, pos, num_atoms, knn=32)
```

### Cutoff graph

```python
from crystallographic_graph import make_graph

edges = make_graph(cell, pos, num_atoms, cutoff=5.0)
```

### Demo on random structures

```python
import time

import torch

from crystallographic_graph import make_graph

device = "cuda"
structure = 64
atoms = 16

num_atoms = torch.full(
    (structure,),
    fill_value=atoms,
    dtype=torch.long,
    device=device,
)
pos = torch.rand(
    (structure * atoms, 3),
    dtype=torch.float32,
    device=device,
)
cell = torch.matrix_exp(
    torch.randn(
        (structure, 3, 3),
        dtype=torch.float32,
        device=device,
    )
)

t0 = time.time()

edges = make_graph(cell, pos, num_atoms, knn=32)

torch.cuda.synchronize(device=device)

t1 = time.time()

print("knn graph")

print("sources indices:")
print(edges.src)

print("destinations indices:")
print(edges.dst)

print("coordinate of the cell of the destinations:")
print(edges.cell)

print(f"duration: {(t1-t0)*1000:.3f} ms")
```

## Requirements

* python 3.8 or greater
* torch
* torch-scatter

## Installation

From gitlab:

```bash
pip install git+https://gitlab.com/chem-test/crystallographic-graph.git 
```

Local installation:

```bash
git clone https://gitlab.com/chem-test/crystallographic-graph.git
cd crystallographic-graph
pip install .
```

## Unit test

```bash
pytest
```

