import numpy as np
import pandas as pd
import pytest

from musical import cluster

PATH = "tests/test_data"
PATH_TEST_DATA = f"{PATH}/cluster"


@pytest.fixture
def squareform():
    return np.load(f"{PATH_TEST_DATA}/squareform.npy")


@pytest.fixture
def cluster_membership():
    return np.load(f"{PATH_TEST_DATA}/cluster_membership.npy")


@pytest.fixture
def within_cluster_variation():
    return np.load(f"{PATH_TEST_DATA}/within_cluster_variation.npy")


def test_within_cluster_variation(
    squareform, cluster_membership, within_cluster_variation
):
    result = cluster._within_cluster_variation(squareform, cluster_membership)
    assert np.allclose(result, within_cluster_variation)
