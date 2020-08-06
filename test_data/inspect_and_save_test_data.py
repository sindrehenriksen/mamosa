import numpy as np
from mamosa.synthetic import SyntheticData
from mamosa.utils.plot_utils import imshow_seismic
from mamosa.utils.save_load_utils import save_synthetic_data

# TODO: Remove data from repo, save on first run of tests.


def generate_test_data(I_3d, X, T, inspect, save, seed=0):
    generate_simple_data(X, T, inspect, save, seed)
    generate_noisy_data(X, T, inspect, save, seed)
    generate_data_with_fault(X, T, inspect, save, seed)
    generate_data_with_non_constant_reflection_coeff(X, T, inspect, save, seed)
    generate_data_with_multiple_horizons_and_noise(X, T, inspect, save, seed)
    generate_3d_data(I_3d, X, T, inspect, save, seed)
    generate_complex_3d_data(I_3d, X, T, inspect, save, seed)


def save_data(name: str, synth: SyntheticData):
    save_synthetic_data("test_data/data/" + name, synth)


def generate_simple_data(X, T, inspect, save, seed=0):
    np.random.seed(seed)
    synth = SyntheticData((1, X, T))
    synth.generate_horizons(1, generate_reflection_coeffs=False)
    reflection_coeff = 0.2
    synth.generate_synthetic_seismic(reflection_coeff)
    if inspect:
        imshow_seismic(synth.seismic[0], synth.horizons)
    if save:
        name = "simple"
        save_data(name, synth)


def generate_noisy_data(X, T, inspect, save, seed=0):
    np.random.seed(seed)
    synth = SyntheticData((1, X, T))
    synth.generate_horizons(1, generate_reflection_coeffs=False)
    reflection_coeff = 0.2
    synth.generate_synthetic_seismic(
        reflection_coeff, systematic_sigma=0.01, white_sigma=0.001, blur_sigma=0.2
    )
    if inspect:
        imshow_seismic(synth.seismic[0], synth.horizons)
    if save:
        name = "noisy"
        save_data(name, synth)


def generate_data_with_fault(X, T, inspect, save, seed=0):
    np.random.seed(seed)
    synth = SyntheticData((1, X, T))
    synth.generate_horizons(1, fault_xlines=5, generate_reflection_coeffs=False)
    reflection_coeff = 0.2
    synth.generate_synthetic_seismic(reflection_coeff)
    if inspect:
        imshow_seismic(synth.seismic[0], synth.horizons)
    if save:
        name = "fault"
        save_data(name, synth)


def generate_data_with_non_constant_reflection_coeff(X, T, inspect, save, seed=0):
    np.random.seed(seed)
    synth = SyntheticData((1, X, T))
    synth.generate_horizons(1)
    synth.generate_synthetic_seismic()
    if inspect:
        imshow_seismic(synth.seismic[0], synth.horizons)
    if save:
        name = "non_constant_reflection_coeff"
        save_data(name, synth)


def generate_data_with_multiple_horizons_and_noise(X, T, inspect, save, seed=0):
    np.random.seed(seed)
    synth = SyntheticData((1, X, T))
    synth.generate_horizons(2, min_distance=3, generate_reflection_coeffs=False)
    reflection_coeffs = [0.2, 0.1]
    synth.generate_synthetic_seismic(
        reflection_coeffs=reflection_coeffs,
        systematic_sigma=0.01,
        white_sigma=0.001,
        blur_sigma=0.01,
    )
    if inspect:
        imshow_seismic(synth.seismic[0])
    if save:
        name = "multiple"
        save_data(name, synth)


def generate_3d_data(I, X, T, inspect, save, seed=0):
    np.random.seed(seed)
    synth = SyntheticData((I, X, T))
    synth.generate_horizons(1, generate_reflection_coeffs=False)
    reflection_coeff = 0.2
    synth.generate_synthetic_seismic(reflection_coeffs=reflection_coeff)
    if inspect:
        for i in range(I):
            imshow_seismic(synth.seismic[i], synth.horizons[0, i])
    if save:
        name = "3d"
        save_data(name, synth)


def generate_complex_3d_data(I, X, T, inspect, save, seed=0):
    np.random.seed(seed)
    synth = SyntheticData((I, X, T))
    synth.generate_horizons(2, min_distance=3, reflection_coeff_seeds=[-0.25, -0.1])
    synth.generate_synthetic_seismic(
        systematic_sigma=0.01, white_sigma=0.001, blur_sigma=0.1
    )
    if inspect:
        for i in range(I):
            imshow_seismic(synth.seismic[i])
    if save:
        name = "complex_3d"
        save_data(name, synth)
    return synth.reflection_coeffs_array
