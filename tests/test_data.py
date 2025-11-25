""" Tests
    Run in home directory 'datalock':
    > python -m pytest

"""
import pytest
# import numpy as np
from numpy.testing import assert_allclose

from datalock.experimentalData import ExperimentalData


@pytest.fixture
def exd():
    """Creates a fresh instance of ExperimentalData before each test."""
    return ExperimentalData()

def test_empty_dataset():
    exd = ExperimentalData()
    assert exd.count == 0
    assert exd.sets == []

def test_always_passes():
    assert True