import pytest
from mamosa.synthetic import SyntheticData
import numpy as np
from numpy.testing import assert_array_equal


@pytest.fixture()
def seed():
    np.random.seed(123)


@pytest.fixture
def horizons():
    horizons = np.array(
        [[[0, 1, 1], [0, 1, 0], [1, 1, -1]], [[1, 2, 2], [-1, -1, -1], [-1, -1, -1]]]
    )
    return horizons


@pytest.mark.filterwarnings("ignore:horizon")
def test_generate_horizons(seed):
    shape = (16, 16, 16)
    synth = SyntheticData(shape)

    n_horizons = 3
    min_dist = 3
    horizons = synth.generate_horizons(n_horizons, min_dist, fault_size=2)
    assert_array_equal(horizons, synth.horizons)
    assert np.all(np.isin(horizons, np.arange(-1, shape[2])))
    assert horizons.shape == (n_horizons, shape[0], shape[1])
    diff = horizons[1:] - horizons[:-1]
    oob = horizons[1:] == -1
    assert np.all(np.logical_or(diff >= min_dist, oob))

    synth.facies = 1
    synth.seismic = 1
    synth.oob_horizons = [1]
    n_horizons = 1
    horizons = synth.generate_horizons(n_horizons, min_dist, fault_size=0)
    assert horizons.shape[0] == n_horizons
    assert synth.facies is None
    assert synth.seismic is None
    assert synth.oob_horizons == []  # This can actually fail by randomness


def test_generate_oob_horizons(seed):
    shape = (16, 16, 16)
    synth = SyntheticData(shape)
    # Generate out of bound horizons
    with pytest.warns(UserWarning, match="horizon"):
        synth.generate_horizons(3, shape[2])
    assert synth.oob_horizons.__len__() > 0
    with pytest.warns(UserWarning, match="horizon"):
        h_vol = synth.horizon_volume(2)
    assert h_vol is None


def test_horizon_volume(horizons):
    synth = SyntheticData((3, 3, 3))
    synth.horizons = horizons
    h_vol = synth.horizon_volume(0)
    h_vol_true = np.array(
        [
            [[1, 0, 0], [0, 1, 0], [0, 1, 0]],
            [[1, 0, 0], [0, 1, 0], [1, 0, 0]],
            [[0, 1, 0], [0, 1, 0], [0, 0, 0]],
        ]
    )
    assert_array_equal(h_vol, h_vol_true)


def test_ixtn_horizons(horizons):
    synth = SyntheticData((3, 3, 3))
    synth.horizons = horizons
    ixtn = synth.ixtn_horizons()
    ixtn_true = np.array(
        [
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 2, 1, 0],
            [1, 0, 0, 0],
            [1, 1, 1, 0],
            [1, 2, 0, 0],
            [2, 0, 1, 0],
            [2, 1, 1, 0],
            [0, 0, 1, 1],
            [0, 1, 2, 1],
            [0, 2, 2, 1],
        ]
    )
    assert_array_equal(ixtn, ixtn_true)


def test_get_facies(horizons):
    synth = SyntheticData((3, 3, 3))
    synth.horizons = horizons
    facies = synth.get_facies()
    facies_true = np.array(
        [
            [[1, 2, 2], [0, 1, 2], [0, 1, 2]],
            [[1, 1, 1], [0, 1, 1], [1, 1, 1]],
            [[0, 1, 1], [0, 1, 1], [0, 0, 0]],
        ]
    )
    assert_array_equal(facies, facies_true)


@pytest.mark.filterwarnings("ignore:horizon")  # ignore out of bounds warning
def test_generate_synthetic_seismic(seed):
    synth = SyntheticData((3, 3, 3))
    synth.generate_horizons(2, 1)
    seismic = synth.generate_synthetic_seismic([0.1, -0.1])
    assert_array_equal(seismic, synth.seismic)
    assert np.all(np.logical_or(seismic <= 1, seismic >= -1))
