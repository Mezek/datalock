""" Tests
    Run in home directory 'bvpasion':
    > python -m pytest

"""
# import numpy as np
from numpy.testing import assert_allclose

from datalock.experimentalData import ExperimentalData


def test_always_passes():
    assert True

def test_uppercase():
    assert "loud noises".upper() == "LOUD NOISES"

def test_reversed():
    assert list(reversed([1, 2, 3, 4])) == [4, 3, 2, 1]
