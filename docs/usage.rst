Usage
=====

Installation
------------

The easiest approach to install is to use pip_.

Installation with `pip <https://pip.pypa.io/en/latest/>`_ and a
*minimal set of dependencies*:

.. code-block:: console

  pip install kda

Source Code
-----------

**Source code** is available from
https://github.com/Becksteinlab/kda under the `GNU General Public License,
Version 3 <https://www.gnu.org/licenses/gpl-3.0.en.html>`_. Obtain the sources
with `git <https://git-scm.com/>`_:

.. code-block:: console

   git clone https://github.com/Becksteinlab/kda.git

Calculating State Probabilities
-------------------------------

KDA has a host of capabilities, all beginning with defining the connections and reaction rates (if desired) for your system. This is done by constructing an ``NxN`` array with diagonal values set to zero, and off-diagonal values ``(i, j)`` representing connections (and reaction rates) between states ``i`` and ``j``. If desired, these can be the edge weights (denoted ``kij``), but they can be specified later.

The following is an example for a simple 3-state model with all nodes connected:

.. code-block:: python

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

The output from the above example:

.. code-block:: console

  $ python example.py
  State probabilities:
   [0.33333333 0.33333333 0.33333333]
  State 1 probability expression:
   (k21*k31 + k21*k32 + k23*k31)/(k12*k23 + k12*k31 + k12*k32 + k13*k21 + k13*k23 + k13*k32 + k21*k31 + k21*k32 + k23*k31)

As expected, the state probabilities are equal because all edge weights are set to a value of 1.

Calculating Transition Fluxes
-----------------------------
Continuing with the previous example, the transition fluxes (one-way or net) can be calculated from the :class:`~kda.core.KineticModel`:

.. code-block:: python

  # make sure the symbolic probabilities have been generated
  model.build_state_probabilities(symbolic=True)
  # iterate over all edges
  print("One-way transition fluxes:")
  for (i, j) in model.G.edges():
      flux = model.get_transition_flux(state_i=i+1, state_j=j+1, net=False, symbolic=True)
      print(f"j_{i+1}{j+1} = {flux}")

The output from the above example:

.. code-block:: console

  $ python example.py
  One-way transition fluxes:
  j_12 = (k12*k21*k31 + k12*k21*k32 + k12*k23*k31)/(k12*k23 + k12*k31 + k12*k32 + k13*k21 + k13*k23 + k13*k32 + k21*k31 + k21*k32 + k23*k31)
  j_13 = (k13*k21*k31 + k13*k21*k32 + k13*k23*k31)/(k12*k23 + k12*k31 + k12*k32 + k13*k21 + k13*k23 + k13*k32 + k21*k31 + k21*k32 + k23*k31)
  j_21 = (k12*k21*k31 + k12*k21*k32 + k13*k21*k32)/(k12*k23 + k12*k31 + k12*k32 + k13*k21 + k13*k23 + k13*k32 + k21*k31 + k21*k32 + k23*k31)
  j_23 = (k12*k23*k31 + k12*k23*k32 + k13*k23*k32)/(k12*k23 + k12*k31 + k12*k32 + k13*k21 + k13*k23 + k13*k32 + k21*k31 + k21*k32 + k23*k31)
  j_31 = (k12*k23*k31 + k13*k21*k31 + k13*k23*k31)/(k12*k23 + k12*k31 + k12*k32 + k13*k21 + k13*k23 + k13*k32 + k21*k31 + k21*k32 + k23*k31)
  j_32 = (k12*k23*k32 + k13*k21*k32 + k13*k23*k32)/(k12*k23 + k12*k31 + k12*k32 + k13*k21 + k13*k23 + k13*k32 + k21*k31 + k21*k32 + k23*k31)


Displaying Diagrams
-------------------

Continuing with the previous example, the KDA ``diagrams`` and ``plotting`` modules can be leveraged to display the diagrams that lead to the above probability expression:

.. code-block:: python

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

This will generate two files, ``input.png`` and ``directional_panel.png``, in your current working directory:

**input.png**

|img_3_input|

**directional_panel.png**

|img_3_directional|

**NOTE:** For more examples (like the following) visit the
`KDA examples <https://github.com/Becksteinlab/kda-examples>`_ repository:

|img_4wl| |img_5wl|
|img_6wl| |img_8wl|
