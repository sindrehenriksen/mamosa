import numpy as np
import bruges
import scipy.stats
import scipy.linalg
import warnings
from scipy.ndimage import gaussian_filter
from typing import Tuple, Union, List, Optional, Callable, Any

# TODO: Add support for horizons that "stop"/"vanish" (i.e. a layer is eroded).


class SyntheticData:
    """Class for generating synthetic geo-volumes and seismic therefrom.

    This class can do the following:
        - Generate semi-realistic random synthetic horizons inn a subsurface volume of
            the desired size (number of voxels). The horizons cover the entire volume.
        - Generate simple (unrealistic), parallel faults.
        - Generate synthetic seismic data from the synthetic subsurface volume.

    Args:
        shape (Tuple[int, int, int]): Shape of the synthetic geo-volume, on the format
            (I, X, T).

    Attributes:
        I: Number of ilines, > 0.
        X: Number of xlines, > 0.
        T: Number of tlines, > 0.
        n_horizons: Number of horizons in geo-volume, > 0.
        horizons: List of length n_horizons of ndarray of int, shape (I, X). Element
            (I, X) of list element h gives the height of horizon h in (I, X) - only one
            horizon point per horizon per trace is supported. -1 indicates out of
            bounds, i.e. the horizon is not in the geo-volume.
        facies: ndarray of int, shape (I, X, T). Facies start at horizons (inclusive)
            and continue to next horizon (exclusive) in t-direction. I.e.
            n_facies = n_horizons + 1. The array contains integers from 0 to n_horizons.
        seismic: ndarray of float, shape (I, X, T). Synthetic seismic.
        wavelet: array_like; list of wavelet amplitudes.
        reflection_coeffs: List of reflection coefficients, one for each horizon. Each
            can be a float (constant coefficients across horizons) or an (I*X) array.
            -1 < reflection coefficient < 1.
        oob_horizons: List of horizons that are partly or entirely out of bounds, i.e.
            some/all points of the horizon not in the geo-volume.

    """

    def __init__(self, shape: Tuple[int, int, int]):
        self.I, self.X, self.T = shape
        self.n_horizons = 0
        self.horizons: Optional[np.ndarray] = None
        self.facies: Optional[np.ndarray] = None
        self.seismic: Optional[np.ndarray] = None
        self.wavelet: Any = None
        self.reflection_coeffs: Optional[np.ndarray] = None
        self.oob_horizons: List[int] = []
        self._systematic_sigma = 0.0
        self._white_sigma = 0.0
        self._blur_sigma = 0.0
        self._systematic_noise: Optional[np.ndarray] = None
        self._white_noise: Optional[np.ndarray] = None
        self._blur_noise: Optional[np.ndarray] = None

    @property
    def shape(self) -> Tuple[int, int, int]:
        """Shape property.

        Returns:
            Tuple[int, int, int]: Shape of geo-volume (I*X*T).

        """
        return self.I, self.X, self.T

    @property
    def reflection_coeffs_array(self) -> Optional[np.ndarray]:
        """Reflection coefficient array property.

        Returns:
            np.ndarray: Shape (I*X*T); array of reflection coefficients.

        """
        if self.reflection_coeffs is None:
            return None
        else:
            r_array = np.zeros(self.shape)
            ii, xx = np.mgrid[: self.I, : self.X]
            for i in range(self.n_horizons):
                h = self.horizons[i]  # type: ignore
                r_array[ii, xx, h] = self.reflection_coeffs[i]
            return r_array

    @property
    def noise(self) -> np.ndarray:
        """Noise property.

        Subtracting noise from self.seismic gives noise-free seismic.

        Returns:
            np.ndarray: Shape (I*X*T); array of noise contribution to seismic.

        """
        if self._blur_noise is not None:
            return self._blur_noise
        if self._systematic_noise is not None:
            if self._white_noise is not None:
                return self._systematic_noise + self._white_noise
            return self._systematic_noise
        if self._white_noise is not None:
            return self._white_noise
        return np.zeros(self.shape)

    def generate_horizons(
        self,
        n_horizons: int,
        min_distance: int = 5,
        volatility: float = 0.6,
        trend_size: float = 1,
        trend_length: int = 30,
        fault_xlines: Union[int, List[int]] = None,
        fault_size: Union[int, List[int]] = 5,
        generate_reflection_coeffs: bool = True,
        reflection_coeff_volatility: float = 0.005,
        reflection_coeff_seeds: List[float] = None,
    ) -> np.ndarray:
        """Generate synthetic horizons.

        Generate random synthetic horizons in the defined synthetic geo-volume.

        Args:
            n_horizons: int > 0. Number of horizons to be generated.
            min_distance: int >= 0. Minimum distance between the horizons (and top
                horizon and 0).
            volatility: float > 0. Decides the volatility of the horizons.
            trend_size: float > 0. Decides how significant trends the horizons have.
            trend_length: float > 0. Decides how long the trends last for.
            fault_xlines: Create faults at these xlines.
            fault_size: List of size of fault jumps, or size of all jumps if just an
                integer. Ignored if fault_xlines is None.
            generate_reflection_coeffs: If True, generate random, non-constant
                reflection coefficients.
            reflection_coeff_volatility: float > 0. Volatility of the reflection
                coefficients.
            reflection_coeff_seeds: Initial values that the random reflection
                coefficients will fluctuate around.

        Returns:
            List of horizon numpy arrays of size (I*X).

        """
        # Reset:
        self.facies = None
        self.seismic = None
        self.oob_horizons = []
        self.n_horizons = n_horizons
        if reflection_coeff_seeds is not None:
            msg = (
                "Please provide a reflection coefficient seed value for each horizon, "
                "if any."
            )
            assert len(reflection_coeff_seeds) == self.n_horizons, msg
            # TODO: Should respect bounds from _generate_horizons.
        self.horizons = self._generate_overlapping_horizons(
            volatility,
            trend_length,
            trend_size,
            generate_reflection_coeffs,
            reflection_coeff_volatility,
            reflection_coeff_seeds,
        )
        self.horizons = self._set_min_distance(min_distance)
        if fault_xlines is not None:
            if isinstance(fault_xlines, int):
                fault_xlines = [fault_xlines]
            if isinstance(fault_size, int):
                fault_size = [fault_size] * len(fault_xlines)
            else:
                assert len(fault_size) == len(fault_xlines)
            for x, size in zip(fault_xlines, fault_size):
                self.horizons = self.create_fault(x, size)
        self.horizons = self._move_above_zero(min_distance)
        self.horizons = self._set_oob()  # set points above top of vol to 0
        return self.horizons

    def _generate_overlapping_horizons(
        self,
        volatility: float,
        trend_length: int,
        trend_size: float,
        generate_reflection_coeffs: bool,
        reflection_coeff_volatility: float,
        reflection_coeff_seeds: Optional[List[float]],
    ) -> np.ndarray:
        """Generate horizons independently. They will overlap."""
        horizons = np.zeros((self.n_horizons, self.I, self.X))
        if generate_reflection_coeffs:
            self.reflection_coeffs = np.zeros((self.n_horizons, self.I, self.X))

        # Create trend vectors
        i_trend = self._get_trend_vec(self.I, trend_size, trend_length)
        x_trend = self._get_trend_vec(self.X, trend_size, trend_length)

        def _jump_r(trend):
            return volatility * np.random.randn() + trend

        # Generate one horizon at a time according to a random process using
        # the trend vectors
        for h in range(0, self.n_horizons):
            horizons[h] = self._generate_horizon(i_trend, x_trend, _jump_r)
        if generate_reflection_coeffs:
            rel_vol = reflection_coeff_volatility / volatility

            def _jump_c(trend):
                return reflection_coeff_volatility * np.random.randn() + rel_vol * trend

            for h in range(0, self.n_horizons):
                # Trend might be decreasing with increasing depth
                flip = np.random.choice((-1, 1))
                if reflection_coeff_seeds is None:
                    seed = None
                else:
                    seed = reflection_coeff_seeds[h]
                self.reflection_coeffs[h] = self._generate_horizon(  # type: ignore
                    flip * i_trend, flip * x_trend, _jump_c, True, seed
                )

        # horizons should be integer-valued.
        horizons = horizons.round().astype(int)

        return horizons

    def _generate_horizon(
        self,
        i_trend: np.ndarray,
        x_trend: np.ndarray,
        jump: Callable,
        reflection_coeff: bool = False,
        reflection_coeff_seed: float = None,
    ) -> np.ndarray:
        """Generate and return a single horizon or horizon reflection coefficients."""
        iline_edge = np.zeros(self.I)
        xline_edge = np.zeros(self.X)
        if reflection_coeff:
            if reflection_coeff_seed is not None:
                iline_edge[0] = reflection_coeff_seed
                xline_edge[0] = reflection_coeff_seed
            else:
                # Init range (-0.25, -0.1) or (0.1, 0.25)
                iline_edge[0] = np.random.uniform(-0.15, 0.15)
                iline_edge[0] += np.sign(iline_edge[0]) * 0.1
                xline_edge[0] = iline_edge[0]
            high = 0.3 * np.sign(iline_edge[0])
            low = 0.05 * np.sign(iline_edge[0])
            if high < low:
                high, low = (low, high)
        else:
            high = np.inf
            low = -high
        # Generate the horizon along the edges iline = 0 and xline = 0.
        for i in range(1, self.I):
            iline_edge[i] = (iline_edge[i - 1] + jump(i_trend[i])).clip(low, high)
        for x in range(1, self.X):
            xline_edge[x] = (xline_edge[x - 1] + jump(x_trend[x])).clip(low, high)
        horizon = np.zeros((self.I, self.X))
        horizon[:, 0] = iline_edge
        horizon[0, :] = xline_edge
        # Generate the rest of the horizon.
        for i in range(1, self.I):
            for x in range(1, self.X):
                i_jump = jump(i_trend[i])
                x_jump = jump(x_trend[x])
                horizon[i, x] = (
                    0.5 * (horizon[i - 1, x] + i_jump + horizon[i, x - 1] + x_jump)
                ).clip(low, high)
        return horizon

    def _get_trend_vec(
        self, n: int, trend_size: float, trend_length: int
    ) -> np.ndarray:
        """Get trend of a random walk with trend."""
        trend = trend_size * np.random.randn(n)
        trend[0] = 0
        trend = self._moving_average(trend, trend_length)
        return trend

    @staticmethod
    def _moving_average(a: np.ndarray, n: int) -> np.ndarray:
        """Moving average of a, window size = n."""
        b = np.copy(a)
        b = np.insert(b, 0, np.full(n, a[0]))
        s = np.cumsum(b)
        res = (s[n:] - s[:-n]) / n
        return res

    def _set_min_distance(self, min_distance: int) -> np.ndarray:
        """Move horizons to fulfill minimum distance specification."""
        for j in range(1, self.n_horizons):
            diff = self.horizons[j] - self.horizons[j - 1]  # type: ignore
            min_diff = diff.min()
            if min_diff < min_distance:
                dist = np.random.randint(min_distance, 3 * min_distance)
                self.horizons[j] += dist - min_diff  # type: ignore
        return self.horizons

    def create_fault(self, fault_xline: int, fault_size: int) -> np.ndarray:
        """Create a fault at a xline fault_xline.

        Args:
            fault_xline: Xline to create fault at.
            fault_size: Size of fault.

        Returns:
            See class attribute self.horizons.

        """
        self.horizons[:, :, fault_xline:] += fault_size  # type: ignore
        return self.horizons

    def _move_above_zero(self, min_distance: int) -> np.ndarray:
        """Make sure that the top horizon is a little above 0 (below seabed)."""
        h_min = self.horizons[0].min()  # type: ignore
        self.horizons -= h_min
        self.horizons += np.random.randint(0, self.T // min(10, self.T))
        self.horizons += min_distance
        return self.horizons

    def _set_oob(self) -> np.ndarray:
        """Remove parts of horizons above (geologically below) defined geo-volume."""
        oob = self.horizons > (self.T - 1)  # type: ignore
        if oob.sum() > 0:  # type: ignore
            self.horizons[oob] = -1  # type: ignore
            for h in range(self.n_horizons - 1, -1, -1):
                n_out = oob[h].sum()  # type: ignore
                if n_out > 0:
                    I, X = self.I, self.X
                    warnings.warn(
                        f"horizon {h} is "
                        f'{"partly" if n_out < (I*X) else "entirely"} '
                        f"out of bounds."
                    )
                    self.oob_horizons.append(h)
                else:
                    break
        return self.horizons

    def horizon_volume(self, horizon_number: int) -> Optional[np.ndarray]:
        """Produce horizon volume for a single horizon.

        This function transforms the generated horizon into a binary numpy array of
        dimensions (I, X, T). The horizon is represented by the ones.

        Args:
            horizon_number: Which horizon to generate volume for.

        Returns:
            binary ndarray of size (I*X*T) if horizon is (partly) within bounds, None
            otherwise.

        """
        horizon = self.ixtn_horizons()
        horizon = horizon[horizon[:, 3] == horizon_number]
        if horizon.size == 0:
            warnings.warn(f"horizon {horizon_number} is not in volume.")
            return None
        horizon_vol = np.zeros(self.shape)
        horizon_vol[horizon[:, 0], horizon[:, 1], horizon[:, 2]] = 1
        return horizon_vol

    def ixtn_horizons(self) -> np.ndarray:
        """Produce horizon coords.

        This function transforms the generated horizons into a numpy array of dimensions
        (n_horizon_points, 4) with rows (I, X, T, n_horizon).

        Returns:
            ndarray of horizon coords; shape (n_horizon_points, 4).

        """
        in_bounds = self.horizons > -1  # type: ignore
        s = in_bounds.sum()  # type: ignore
        ixtn = np.empty(shape=(s, 4), dtype=int)
        nix = np.argwhere(in_bounds)
        ixtn[:, :2] = nix[:, 1:]
        ixtn[:, 3] = nix[:, 0]
        ixtn[:, 2] = self.horizons[nix[:, 0], nix[:, 1], nix[:, 2]]  # type: ignore
        return ixtn

    def get_facies(self) -> np.ndarray:
        """Generate facies array.

        Returns:
            ndarray of int, shape (I, X, T). See class attribute docstring (facies) for
            description.

        """
        ixtn = self.ixtn_horizons()
        facies = np.zeros(self.shape, dtype=int)

        facies[ixtn[:, 0], ixtn[:, 1], ixtn[:, 2]] = 1
        for t in range(1, self.T):
            facies[:, :, t] = facies[:, :, t] + facies[:, :, (t - 1)]
        self.facies = facies
        return facies

    def generate_synthetic_seismic(
        self,
        reflection_coeffs: Union[float, List[Union[float, np.ndarray]]] = None,
        systematic_sigma: float = 0,
        white_sigma: float = 0,
        blur_sigma: float = 0,
        wavelet_frequency: int = 40,
    ):
        """Generate synthetic seismic.

        Create synthetic seismic using instance horizons and coefficients, or provided
        (constant) coefficients.

        Args:
            reflection_coeffs: See class attributes.
            systematic_sigma: Systematic noise added if not None; higher means more
                noise.
            white_sigma: White noise added if not None; higher means more noise.
            blur_sigma: Seismic blurred if not None; higher means more blurred.
            wavelet_frequency: Frequency of wavelet passed to bruges.filters.ricker() to
                define wavelet.

        Returns:
            ndarray of float, shape (I, X, T).

        """
        if reflection_coeffs is not None:
            if isinstance(reflection_coeffs, float):
                self.reflection_coeffs = np.array(reflection_coeffs).reshape(1)
            else:
                self.reflection_coeffs = np.array(reflection_coeffs)
            msg = (
                "Please provide one reflection coefficient constant/array for each"
                "horizon."
            )
            assert len(self.reflection_coeffs) == self.n_horizons, msg
            assert np.all(np.abs(self.reflection_coeffs) < 1), "Max 100% reflected."
        if self.reflection_coeffs is None:
            warnings.warn("No reflection coefficients. Cannot generate seismic.")
            return

        dt = 0.005
        # For some reason, odd length of the wave gives two spike points, we want one...
        even_T = self.T - self.T % 2
        duration = min(0.100, 0.005 * even_T)  # n_steps <= self.T
        wave = bruges.filters.ricker(duration=duration, dt=dt, f=wavelet_frequency)
        # ... but we want odd length
        wave = np.delete(wave, 0)
        self.wavelet = wave
        # TODO: Quicker to use convolution_matrix here?
        reflection_arr = self.reflection_coeffs_array
        seismic = np.apply_along_axis(
            lambda r: np.convolve(r, wave, mode="same"), axis=-1, arr=reflection_arr
        )
        self.seismic = seismic

        if systematic_sigma > 0:
            first_col = np.zeros(self.T)
            l = wave.size // 2 + 1
            first_col[:l] = wave[(l - 1) :]
            convolution_matrix = scipy.linalg.toeplitz(first_col)
            self._systematic_sigma = systematic_sigma
            W = convolution_matrix
            covariance_matrix = systematic_sigma ** 2 * W @ W.T
            dist = scipy.stats.multivariate_normal(np.zeros(self.T), covariance_matrix)
            self._systematic_noise = dist.rvs((self.I, self.X))
            seismic += self._systematic_noise
        else:
            self._systematic_sigma = 0
        if white_sigma > 0:
            self._white_sigma = white_sigma
            self._white_noise = np.random.normal(np.zeros(seismic.shape), white_sigma)
            seismic += self._white_noise
        else:
            self._white_sigma = 0
        if blur_sigma > 0:
            self._blur_sigma = blur_sigma
            seismic = gaussian_filter(seismic, sigma=[blur_sigma, blur_sigma, 0])
            self._blur_noise = self.seismic - seismic
        else:
            self._blur_sigma = 0

        self.seismic = seismic
        return seismic
