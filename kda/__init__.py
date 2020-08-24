"""
Kinetic Diagram Analysis
Python package used for the analysis of biochemical kinetic diagrams.
"""

# Add imports here
from kda.core import *

# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
