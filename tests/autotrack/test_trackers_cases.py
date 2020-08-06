from pytest_cases import test_target
from mamosa.utils.save_load_utils import load_synthetic_data

# TODO: Data as params?


def load_data(name: str):
    return load_synthetic_data("../test_data/data/" + name)


i, x = 0, 10
synth = load_data("simple")


def case_simple():
    seed_point = (i, x, synth.horizons[0, i, x])
    return (
        (synth.seismic, synth.wavelet, seed_point, synth.reflection_coeffs),
        (synth.horizons[0], None),
        None,
    )


def case_negative():
    seed_point = (i, x, synth.horizons[0, i, x])
    return (
        (-synth.seismic, synth.wavelet, seed_point, -synth.reflection_coeffs),
        (synth.horizons[0], None),
        None,
    )


def case_multiple_seeds():
    i2, x2 = i, x + 5
    seed_points = ((i, x, synth.horizons[0, i, x]), (i2, x2, synth.horizons[0, i2, x2]))
    return (
        (
            synth.seismic,
            synth.wavelet,
            seed_points,
            [synth.reflection_coeffs] * len(seed_points),
        ),
        (synth.horizons[0], None),
        None,
    )


synth_noise = load_data("noisy")


def case_noise():
    seed_point = (i, x, synth_noise.horizons[0, i, x])
    return (
        (
            synth_noise.seismic,
            synth_noise.wavelet,
            seed_point,
            synth_noise.reflection_coeffs,
        ),
        (synth_noise.horizons[0], None),
        None,
    )


synth_fault = load_data("fault")


def case_fault():
    seed_point = (i, x, synth_fault.horizons[0, i, x])
    return (
        (
            synth_fault.seismic,
            synth_fault.wavelet,
            seed_point,
            synth_fault.reflection_coeffs,
        ),
        (synth_fault.horizons[0], None),
        None,
    )


synth_non_const_refl = load_data("non_constant_reflection_coeff")


@test_target("reflection_coefficients")
def case_non_const_reflection_coeff():
    seed_point = (i, x, synth_non_const_refl.horizons[0, i, x])
    return (
        (
            synth_non_const_refl.seismic,
            synth_non_const_refl.wavelet,
            seed_point,
            synth_non_const_refl.reflection_coeffs[0, i, x],
        ),
        (synth_non_const_refl.horizons[0], synth_non_const_refl.reflection_coeffs[0]),
        None,
    )


synth_multiple_horizons = load_data("multiple")


def case_multiple_horizons_and_noise():
    seed_points = (
        (i, x, synth_multiple_horizons.horizons[0, i, x]),
        (i, x, synth_multiple_horizons.horizons[1, i, x]),
    )
    return (
        (
            synth_multiple_horizons.seismic,
            synth_non_const_refl.wavelet,
            seed_points,
            synth_multiple_horizons.reflection_coeffs,
        ),
        (synth_multiple_horizons.horizons, None),
        None,
    )


synth_3d = load_data("3d")


@test_target("reflection_coefficients")
def case_3d():
    seed_points = (
        (i, x, synth_3d.horizons[0, i, x]),
        (i + 1, x + 1, synth_3d.horizons[0, i + 1, x + 1]),
    )
    return (
        (
            synth_3d.seismic,
            synth_3d.wavelet,
            seed_points,
            [synth_3d.reflection_coeffs] * len(seed_points),
        ),
        (synth_3d.horizons[0], synth_3d.reflection_coeffs[0]),
        None,
    )


synth_complex = load_data("complex_3d")


def case_3d_complex():
    seed_points = (
        (i, x, synth_complex.horizons[0, i, x]),
        (i + 1, x + 1, synth_complex.horizons[1, i + 1, x + 1]),
    )
    reflection_coeff_seeds = (
        synth_complex.reflection_coeffs[0][seed_points[0][:2]],
        synth_complex.reflection_coeffs[1][seed_points[1][:2]],
    )
    return (
        (
            synth_complex.seismic,
            synth_complex.wavelet,
            seed_points,
            reflection_coeff_seeds,
        ),
        (synth_complex.horizons, synth_complex.reflection_coeffs),
        None,
    )
