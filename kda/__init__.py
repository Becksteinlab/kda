# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
#
# Author: Nikolaus C. Awtrey
#
"""
Core functions of Kinetic Diagram Analysis
=========================================================================

The core class is a :class:`~kda.core.KineticModel` which contains
all the system information (kinetic diagram, transition rates, etc.).

To get started, load the KineticModel::

  model = KineticModel(K=rate_matrix, G=nx.MultiDiGraph)

With the model created the state probability expressions can be
generated using the built-in methods::

  model.build_state_probabilities(symbolic=True)

Other methods are available to generate the various graphs or
calculate fluxes.

"""

# Add imports here
from .core import KineticModel

# Handle versioneer
from ._version import get_versions

versions = get_versions()
__version__ = versions["version"]
__git_revision__ = versions["full-revisionid"]
del get_versions, versions
