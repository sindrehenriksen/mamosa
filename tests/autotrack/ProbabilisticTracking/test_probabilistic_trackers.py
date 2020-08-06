import pytest
import numpy as np
from mamosa.autotrack.ProbabilisticTracking import MAPTracker, GreedyTracker
from mamosa.synthetic import SyntheticData
from numpy.testing import assert_array_equal, assert_array_almost_equal
from pytest_cases import cases_data
from tests.autotrack import test_trackers_cases

# TODO: Test seed point conversion.


def has_tag_refl(tags):
    return "reflection_coefficients" in tags


@pytest.mark.parametrize("Tracker", [MAPTracker, GreedyTracker])
@pytest.mark.parametrize("marginal_likelihood_method", ["local", "full"])
@pytest.mark.parametrize("increasing_var", [True, False])
@cases_data(module=test_trackers_cases)
def test_tracker(Tracker, marginal_likelihood_method, increasing_var, case_data):
    ins, outs, _ = case_data.get()
    seismic, wavelet, seed_points, reflection_coeff_seeds = ins
    horizons_true, _ = outs
    dT = seismic.shape[2]
    tracker = Tracker(
        seismic,
        wavelet,
        marginal_likelihood_method=marginal_likelihood_method,
        increasing_var=increasing_var,
    )
    if horizons_true.ndim == 2:
        horizon = tracker.track_horizon(
            seed_points, reflection_coeff_seeds, start_t=0, dT=dT
        )
        assert_array_equal(horizon, horizons_true)
    else:
        horizon0 = tracker.track_horizon(
            seed_points[0], reflection_coeff_seeds[0], start_t=0, dT=dT
        )
        horizon1 = tracker.track_horizon(
            seed_points[1], reflection_coeff_seeds[1], start_t=0, dT=dT
        )
        assert_array_equal(np.array((horizon0, horizon1)), horizons_true)


@pytest.mark.parametrize("Tracker", [MAPTracker, GreedyTracker])
@pytest.mark.parametrize("marginal_likelihood_method", ["local", "full"])
@pytest.mark.parametrize("increasing_var", [True, False])
@cases_data(module=test_trackers_cases, filter=has_tag_refl)
def test_track_reflection_coeffs(
    Tracker, marginal_likelihood_method, increasing_var, case_data
):
    ins, outs, _ = case_data.get()
    seismic, wavelet, seed_points, reflection_coeff_seeds = ins
    _, reflection_coeffs_true = outs
    dT = seismic.shape[2]
    tracker = Tracker(
        seismic,
        wavelet,
        marginal_likelihood_method=marginal_likelihood_method,
        increasing_var=increasing_var,
    )
    _ = tracker.track_horizon(seed_points, reflection_coeff_seeds, start_t=0, dT=dT)
    assert_array_almost_equal(
        tracker.reflection_coefficients[0], reflection_coeffs_true, 2
    )


@pytest.fixture
def flat_horizon_data():
    I, X, T = (2, 20, 20)
    synth = SyntheticData((I, X, T))
    synth.generate_horizons(1)
    t = T // 2
    synth.horizons[:] = t
    reflection_coeff = 0.2
    synth.generate_synthetic_seismic(reflection_coeff)
    return synth, X, T, t, reflection_coeff


@pytest.mark.parametrize("Tracker", [MAPTracker, GreedyTracker])
@pytest.mark.parametrize("marginal_likelihood_method", ["local", "full"])
@pytest.mark.parametrize("increasing_var", [True, False])
def test_likelihood_method(
    flat_horizon_data, Tracker, marginal_likelihood_method, increasing_var
):
    synth, X, T, t, reflection_coeff = flat_horizon_data
    tracker = Tracker(
        synth.seismic, synth.wavelet, marginal_likelihood_method, increasing_var
    )
    _ = tracker.track_horizon((0, 0, t), reflection_coeff, start_t=0, dT=T)
    assert np.argmax(tracker.get_marginal_likelihoods(0, 0, X // 2)) == t


@pytest.mark.parametrize("Tracker", [MAPTracker, GreedyTracker])
@pytest.mark.parametrize("marginal_likelihood_method", ["local", "full"])
@pytest.mark.parametrize("increasing_var", [True, False])
def test_tracking_in_subset_of_depths(
    flat_horizon_data, Tracker, marginal_likelihood_method, increasing_var
):
    synth, X, T, t, reflection_coeff = flat_horizon_data
    tracker = Tracker(
        synth.seismic, synth.wavelet, marginal_likelihood_method, increasing_var
    )
    start_t = t - 5
    dT = min(t - 1, T - start_t)
    horizon = tracker.track_horizon((0, 0, t), reflection_coeff, start_t=start_t, dT=dT)
    assert np.all(horizon == t)
