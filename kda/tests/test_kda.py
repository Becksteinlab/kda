"""
Unit and regression test for the kda package.
"""

# Import package, test suite, and other packages as needed
import kda
import pytest
import sys

def test_kda_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "kda" in sys.modules
