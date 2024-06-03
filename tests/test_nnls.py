import numpy as np
import pandas as pd
import pytest

from musical import nnls

PATH = "tests/test_data"
PATH_TEST_DATA = f"{PATH}/nnls"


@pytest.fixture
def data_mat():
    data = pd.read_csv(f"{PATH_TEST_DATA}/data.csv", index_col=0)
    return data.values


@pytest.fixture
def signatures_mat():
    signatures = pd.read_csv(f"{PATH_TEST_DATA}/catalog.csv", index_col=0)
    return signatures.values


@pytest.fixture
def exposures_mat():
    return np.load(f"{PATH_TEST_DATA}/exposures_mat.npy")


def test_nnls(data_mat, signatures_mat, exposures_mat):
    result = nnls.nnls(data_mat, signatures_mat)
    assert np.allclose(result, exposures_mat)
