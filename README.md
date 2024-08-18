Kinetic Diagram Analysis
==============================
[//]: # (Badges)
[![CI](https://github.com/Becksteinlab/kda/actions/workflows/test.yml/badge.svg)](https://github.com/Becksteinlab/kda/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/Becksteinlab/kda/branch/master/graph/badge.svg)](https://codecov.io/gh/Becksteinlab/kda/branch/master)
[![Documentation Status](https://readthedocs.org/projects/kda/badge/?version=latest)](https://kda.readthedocs.io/en/latest/?badge=latest)
[![asv](http://img.shields.io/badge/benchmarked%20by-asv-blue.svg?style=flat)](https://github.com/Becksteinlab/kda/actions/workflows/test.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5826394.svg)](https://doi.org/10.5281/zenodo.5826394)

Python package used for the analysis of biochemical kinetic diagrams using the diagrammatic approach developed by [T.L. Hill](https://link.springer.com/book/10.1007/978-1-4612-3558-3).

**WARNING:** this software is in flux and is not API stable.

## Examples

KDA has a host of capabilities, all beginning with defining the connections and reaction rates (if desired) for your system. This is done by constructing an `NxN`array with diagonal values set to zero, and off-diagonal values `(i, j)` representing connections (and reaction rates) between states `i` and `j`. If desired, these can be the edge weights (denoted `kij`), but they can be specified later.

The following is an example for a simple 3-state model with all nodes connected:
```python
import numpy as np
import kda

# define matrix with reaction rates set to 1
K = np.array(
    [
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0],
    ]
)
# create a KineticModel from the rate matrix
model = kda.KineticModel(K=K, G=None)
# get the state probabilities in numeric form
model.build_state_probabilities(symbolic=False)
print("State probabilities: \n", model.probabilities)
# get the state probabilities in expression form
model.build_state_probabilities(symbolic=True)
print("State 1 probability expression: \n", model.probabilities[0])
```

The output from the above example:
```bash
$ python example.py
State probabilities:
 [0.33333333 0.33333333 0.33333333]
State 1 probability expression:
 (k21*k31 + k21*k32 + k23*k31)/(k12*k23 + k12*k31 + k12*k32
    + k13*k21 + k13*k23 + k13*k32 + k21*k31 + k21*k32 + k23*k31)
```
As expected, the state probabilities are equal because all edge weights are set to a value of 1.

Additionally, the transition fluxes (one-way or net) can be calculated from the `KineticModel`:
```python
# make sure the symbolic probabilities have been generated
model.build_state_probabilities(symbolic=True)
# iterate over all edges
print("One-way transition fluxes:")
for (i, j) in model.G.edges():
    flux = model.get_transition_flux(state_i=i+1, state_j=j+1, net=False, symbolic=True)
    print(f"j_{i+1}{j+1} = {flux}")
```

The output from the above example:
```bash
$ python example.py
One-way transition fluxes:
j_12 = (k12*k21*k31 + k12*k21*k32 + k12*k23*k31)/(k12*k23 + k12*k31 + k12*k32 + k13*k21 + k13*k23 + k13*k32 + k21*k31 + k21*k32 + k23*k31)
j_13 = (k13*k21*k31 + k13*k21*k32 + k13*k23*k31)/(k12*k23 + k12*k31 + k12*k32 + k13*k21 + k13*k23 + k13*k32 + k21*k31 + k21*k32 + k23*k31)
j_21 = (k12*k21*k31 + k12*k21*k32 + k13*k21*k32)/(k12*k23 + k12*k31 + k12*k32 + k13*k21 + k13*k23 + k13*k32 + k21*k31 + k21*k32 + k23*k31)
j_23 = (k12*k23*k31 + k12*k23*k32 + k13*k23*k32)/(k12*k23 + k12*k31 + k12*k32 + k13*k21 + k13*k23 + k13*k32 + k21*k31 + k21*k32 + k23*k31)
j_31 = (k12*k23*k31 + k13*k21*k31 + k13*k23*k31)/(k12*k23 + k12*k31 + k12*k32 + k13*k21 + k13*k23 + k13*k32 + k21*k31 + k21*k32 + k23*k31)
j_32 = (k12*k23*k32 + k13*k21*k32 + k13*k23*k32)/(k12*k23 + k12*k31 + k12*k32 + k13*k21 + k13*k23 + k13*k32 + k21*k31 + k21*k32 + k23*k31)
```

Continuing with the previous example, the KDA `plotting` module can be leveraged to display the diagrams that lead to the above probability expression:
```python
import os
from kda import plotting

# generate the directional diagrams
model.build_directional_diagrams()
# get the current working directory
cwd = os.getcwd()
# specify the positions of all nodes in NetworkX fashion
node_positions = {0: [0, 1], 1: [-0.5, 0], 2: [0.5, 0]}
# plot and save the input diagram
plotting.draw_diagrams(model.G, pos=node_positions, path=cwd, label="input")
# plot and save the directional diagrams as a panel
plotting.draw_diagrams(
    model.directional_diagrams,
    pos=node_positions,
    path=cwd,
    cbt=True,
    label="directional_panel",
)
```

This will generate two files, `input.png` and `directional_panel.png`, in your current working directory:

#### `input.png`
<img src="https://github.com/Becksteinlab/kda-examples/blob/master/kda_examples/test_model_3_state/diagrams/input.png" width=300, alt="3-state model input diagram">

#### `directional_panel.png`
<img src="https://github.com/Becksteinlab/kda-examples/blob/master/kda_examples/test_model_3_state/diagrams/directional.png" width=300, alt="3-state model directional diagrams">

**NOTE:** For more examples (like the following) visit the [KDA examples](https://github.com/Becksteinlab/kda-examples) repository:

<img src="https://github.com/Becksteinlab/kda-examples/blob/master/kda_examples/test_model_4_state_leakage/diagrams/input.png" width=250, alt="4-state model with leakage input diagram"> <img src="https://github.com/Becksteinlab/kda-examples/blob/master/kda_examples/test_model_5_state_leakage/diagrams/input.png" width=250, alt="5-state model with leakage input diagram"> <img src="https://github.com/Becksteinlab/kda-examples/blob/master/kda_examples/test_model_6_state_leakage/diagrams/input.png" width=250, alt="6-state model with leakage input diagram">

## Installation
### Development version from source

To install the latest development version from source, run
```bash
git clone git@github.com:Becksteinlab/kda.git
cd kda
python setup.py install
```

## Copyright

Copyright (c) 2020, Nikolaus Awtrey

## Acknowledgements

Project based on the
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.2.
