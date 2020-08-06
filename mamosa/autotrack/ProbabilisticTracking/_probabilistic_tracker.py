import numpy as np
from typing import Any, Callable, List, Optional
import scipy.linalg
import scipy.stats

# TODO: Support more flexibility for estimating reflection coefficients:
#  - control parameters of prior,
#  - control parameters determining domain of reflection coefficients.
# TODO: marginal_likelihood_method=="full" obsolete? "local" seems to both be faster and
#  give better results, in general.
# TODO: increasing_var==True obsolete? - particularly in combination with
#  marginal_likelihood_method=="local" (see above)?


class ProbabilisticTracker:
    """Abstract probability based autotracker base class.

    An autotracker tracks a horizon in seismic data from one or more seed points. The
    seed points are manually chosen points in the horizon which are considered as
    ground truth.

    Args:
        seismic: ndarray of shape (I, X, T); seismic geovolume.
        wavelet: 1d ndarray; the seismic wavelet used in the survey. It is assumed that
            the wavelet is symmetrical and that the largest spike is in the middle. This
            assumptions is important in the self._get_convolution_matrix method.
        marginal_likelihood_method: Calculate marginal likelihoods based on entire
            traces if "full", locally if "local".
        increasing_var: Variance in the likelihood distribution increases with distance
            to the depth considered. The hypothesis is that this decreases significance
            of spikes in the data from other horizons.

    TODO: Document attributes.

    """

    def __init__(
        self,
        seismic: np.ndarray,
        wavelet: np.ndarray,
        marginal_likelihood_method: str,
        increasing_var: bool,
    ) -> None:
        if seismic.ndim == 3:
            self.seismic = seismic
        else:
            assert seismic.ndim == 2
            self.seismic = seismic.reshape(1, *seismic.shape)
        self.cleaned_seismic = np.copy(seismic)
        self.wavelet = wavelet
        self.marginal_likelihood_method = marginal_likelihood_method
        self.increasing_var = increasing_var
        T = self.seismic.shape[2]
        assert wavelet.size <= T
        assert wavelet.ndim == 1
        w_argmax = wavelet.argmax()
        if wavelet.size % 2 == 0:
            assert w_argmax == wavelet.size / 2 or w_argmax == wavelet.size / 2 - 1
        else:
            assert w_argmax == wavelet.size // 2
        if self.marginal_likelihood_method == "local":
            self._set_cut_wavelet(wavelet)
            self._marginal_likelihood = self._local_marginal_likelihood
        elif self.marginal_likelihood_method == "full":
            self._cut_wavelet = None
            self._marginal_likelihood = self._full_marginal_likelihood
        else:
            raise ValueError("marginal_likelihood_method must be 'full' or 'local'.")
        self.seed_points_list: List[np.ndarray] = []
        self.reflection_coefficient_seeds: List[np.ndarray] = []
        self.margins: List[Optional[int]] = []
        self.sigmas: List[float] = []
        self.start_ts: List[int] = []
        self.dTs: List[int] = []
        self.dfs: List[int] = []
        self.ks: List[float] = []
        self._convolution_matrices: List[np.ndarray] = []
        self.horizons: List[np.ndarray] = []
        self.reflection_coefficients: List[np.ndarray] = []
        self._min_refl_coeff_jump = 0.01
        # Reflection coefficients can jump to two levels higher or lower
        self._refl_n_new = 4
        # We need to keep track of reflection coefficient domain for each inline
        self._temp_refl_coeff_domain: Optional[np.ndarray] = None

    def _set_cut_wavelet(self, wavelet: np.ndarray) -> None:
        """Cut away close to zero tails of wavelet.

        Return smallest sub-vector of wavelet accounting for >= 90% of the integral of
        the absolute value of the wavelet. Maximum length, though, is 9.

        """
        w_sum = np.abs(wavelet).sum()
        w_argmax = wavelet.argmax()
        i = 0
        cut_wavelet = wavelet[w_argmax]
        percent_of_abs_integral = np.abs(cut_wavelet) / w_sum
        while percent_of_abs_integral < 0.90 and i < 4:
            i += 1
            if w_argmax - i == -1 or w_argmax + i == wavelet.size:
                i -= 1
                cut_wavelet = wavelet[w_argmax - i : w_argmax + i + 1]
                break
            cut_wavelet = wavelet[w_argmax - i : w_argmax + i + 1]
            percent_of_abs_integral = np.abs(cut_wavelet).sum() / w_sum
        self._cut_wavelet = cut_wavelet

    def track_horizon(
        self,
        seed_points: Any,
        reflection_coefficient_seeds: Any = None,
        horizon_n: int = None,
        margin: int = None,
        start_t: int = None,
        dT: int = None,
        sigma: float = 0.01,
        df: int = 5,
        k: float = 1,
    ) -> np.ndarray:
        """Track seismic horizon.

        Args:
            seed_points: List/array of seed points which are two-tuples if seismic data
                is 2d or three-tuples if seismic data is 3d.
            reflection_coefficient_seeds: Optional. List/array of reflection
                coefficients at the seed points.
            horizon_n: Use to replace horizon estimate. Horizon number corresponds to
                position (depthwise) relative to the other tracked horizons; larger
                means deeper.
            margin: Max distance in time/depth from seed points to potential horizon
                points. Ignored if start_t and dT are given.
            start_t: Starting depth (minimum depth) to consider when tracking horizon.
            dT: Number of tlines to consider when tracking horizon.
            sigma: Higher values means there is more noise in the data. More precisely;
                the data in a trace is assumed to be a Gauss-linear function of the
                reflection coefficient, with covariance matrix sigma*I.
            df: Degrees of freedom used in t-distribution core in clique potentials
                TODO: Ref to report.
            k: Second variable in t-distribution core in clique potentials. Higher k
                means lower probability of large jumps.

        Returns:
            2d horizon depths array, shape (I, X).

        Notes:
            Note that the complexity of the tracking algorithm with respect to dT is
            O(dT^2), so it is important to keep dT as small as possible.

        """
        horizon_n = self._initialize_tracking(
            horizon_n,
            seed_points,
            reflection_coefficient_seeds,
            margin,
            start_t,
            dT,
            sigma,
            df,
            k,
        )
        horizon = self._track(horizon_n)
        self.cleaned_seismic -= self._get_horizon_seismic(horizon_n)
        return horizon

    def _initialize_tracking(
        self,
        horizon_n: Optional[int],
        seed_points: Any,
        reflection_coeff_seeds: Any,
        margin: Optional[int],
        start_t: Optional[int],
        dT: Optional[int],
        sigma: float,
        df: int,
        k: float,
    ) -> int:
        """Set parameters connected to tracking."""
        seed_points = self._numpyfy(seed_points)
        msg = "Seed points must be within seismic cube dimensions."
        assert np.all(seed_points >= 0) and np.all(
            seed_points < self.seismic.shape
        ), msg
        if horizon_n is None:
            horizon_n = self._find_horizon_n(seed_points)
        else:
            assert horizon_n >= 0, "Horizon number must be 0 or larger."
            est = self._find_horizon_n(seed_points)
            msg = (
                "Horizon number must match depth (must be n'th tracked horizon "
                "from top)"
            )
            assert horizon_n in (est - 1, est, est + 1), msg
            if horizon_n < len(self.horizons):
                self._clean_horizon(horizon_n)
        self.seed_points_list.insert(horizon_n, seed_points)
        if reflection_coeff_seeds is not None:
            reflection_coeff_seeds = np.array(reflection_coeff_seeds).reshape(-1)
            msg = (
                "Please provide one reflection coefficient seed for each seed "
                "point, or none at all."
            )
            assert len(reflection_coeff_seeds) == len(seed_points), msg
        self.reflection_coefficient_seeds.insert(horizon_n, reflection_coeff_seeds)
        self._initialize_reflection_coefficients(horizon_n)
        if margin is None and (start_t is None or dT is None):
            margin = 100
        self.margins.insert(horizon_n, margin)
        self.sigmas.insert(horizon_n, sigma)
        self.dfs.insert(horizon_n, df)
        self.ks.insert(horizon_n, k)
        T = self.seismic.shape[2]
        if start_t is None:
            n_seeds = seed_points.shape[0]
            assert n_seeds > 0, "You must set start_t if no seed points are given."
            start_t = seed_points[:, -1].min() - margin
            start_t = max(0, start_t)
        self.start_ts.insert(horizon_n, start_t)
        if dT is None:
            n_seeds = seed_points.shape[0]
            assert n_seeds > 0, "You must set dT if no seed points are given."
            max_t = seed_points[:, -1].max() + margin
            max_t = min(T - 1, max_t)
            dT = max_t - start_t + 1
        self.dTs.insert(horizon_n, dT)
        if self.marginal_likelihood_method == "local":
            wave = self._cut_wavelet
            conv_matrix = self._get_convolution_matrix(wave.size, wave)  # type: ignore
        else:
            conv_matrix = self._get_convolution_matrix(dT, self.wavelet)
        self._convolution_matrices.insert(horizon_n, conv_matrix)
        return horizon_n

    def _numpyfy(self, seed_points: Any) -> np.ndarray:
        """Return seed points as a numpy array of shape (#seeds * 3)."""
        if seed_points is None:
            msg = "All horizons after the first need at least one seed point."
            assert len(self.seed_points_list) == 0, msg
            return np.array([]).reshape(0, 3)
        seed_points = np.array(seed_points)
        if seed_points.ndim == 1:
            seed_points = seed_points.reshape(1, -1)
        else:
            assert seed_points.ndim == 2
        assert seed_points.shape[1] == 3
        assert np.all(seed_points < self.seismic.shape)
        return seed_points

    def _find_horizon_n(self, seed_points: Any) -> int:
        """Return the horizon number, counting from top (0) to bottom (T)."""
        if len(self.seed_points_list) == 0:
            return len(self.horizons)
        i, x = seed_points[:, :2].T
        horizon_n = 0
        for horizon in self.horizons:
            if np.all(horizon[i, x] <= seed_points[:, 2]):
                horizon_n += 1
            elif np.any(horizon[i, x] < seed_points[:, 2]):
                raise AssertionError("Horizons can't cross each other.")
            elif np.all(horizon[i, x] == seed_points[:, 2]):
                raise AssertionError("All seed points are in another horizon.")
            else:
                break
        return horizon_n

    def _clean_horizon(self, horizon_n: int):
        """Clean horizon data when replacing with another estimate."""
        # TODO: Create tests.
        self.cleaned_seismic += self._get_horizon_seismic(horizon_n)
        _ = self.horizons.pop(horizon_n)
        _ = self.seed_points_list.pop(horizon_n)
        _ = self.reflection_coefficients.pop(horizon_n)
        _ = self.reflection_coefficient_seeds.pop(horizon_n)
        _ = self.margins.pop(horizon_n)
        _ = self.sigmas.pop(horizon_n)
        _ = self.start_ts.pop(horizon_n)
        _ = self.dTs.pop(horizon_n)
        _ = self.dfs.pop(horizon_n)
        _ = self.ks.pop(horizon_n)

    def _initialize_reflection_coefficients(self, horizon_n: int):
        """Set initial reflection coefficients for initial iline.

        The initial reflection coefficient guesses are found by interpolating seed
        values.

        """
        # TODO: Create tests.
        I, X, _ = self.seismic.shape
        reflection_coeffs = np.zeros((I, X))
        seed_ix = self.seed_points_list[horizon_n][:, :2]
        i = seed_ix[0, 0]
        indices = np.argwhere(seed_ix[:, 0] == i).reshape(-1)
        refl_seeds = self.reflection_coefficient_seeds[horizon_n]
        if refl_seeds is not None:
            for k in indices:
                j, y = seed_ix[k]
                reflection_coeffs[j, y] = refl_seeds[k]
        else:
            seed_t = self.seed_points_list[horizon_n][:, 2]
            wave_max = self.wavelet[np.abs(self.wavelet).argmax()]
            for k in indices:
                j, y = seed_ix[k]
                s = seed_t[k]
                reflection_coeffs[j, y] = self.seismic[j, y, s] / wave_max * 2
        seed_x = np.sort(seed_ix[indices, 1].reshape(-1))
        reflection_coeffs[i, 0 : seed_x[0]] = reflection_coeffs[i, seed_x[0]]
        for k in range(1, len(seed_x)):
            x_prev = seed_x[k - 1]
            x_next = seed_x[k]
            reflection_coeffs[i, x_prev : x_next + 1] = np.linspace(
                reflection_coeffs[i, x_prev],
                reflection_coeffs[i, x_next],
                x_next - x_prev + 1,
            )
        reflection_coeffs[i, seed_x[-1] :] = reflection_coeffs[i, seed_x[-1]]
        # For greedy method, initialize reflection coefficients outside initial iline
        indices_not_initial_iline = np.argwhere(seed_ix[:, 0] != i).reshape(-1)
        for k in indices_not_initial_iline:
            j, y = seed_ix[k]
            reflection_coeffs[j, y] = refl_seeds[k]
        self.reflection_coefficients.insert(horizon_n, reflection_coeffs)

    @staticmethod
    def _get_convolution_matrix(dT: int, wavelet: np.ndarray) -> np.ndarray:
        """Get convolution matrix C; C * reflection coeff = noise-free seismic."""
        first_col = np.zeros(dT)
        max_spike = np.argmax(np.abs(wavelet))
        l = wavelet.size - max_spike
        l = min(l, dT)
        first_col[:l] = wavelet[max_spike : (max_spike + l)]
        return scipy.linalg.toeplitz(first_col)

    def _track(self, horizon_n: int) -> np.ndarray:
        """Abstract method; track the horizon using the subclass specific method."""
        pass

    def get_marginal_likelihoods(
        self,
        horizon_n: int = 0,
        i: Any = None,
        x: Any = None,
        t: Any = None,
        relative: bool = False,
    ) -> np.ndarray:
        """Return marginal likelihoods.

        i, x, t are used as indices in an array. t is relative to start_t and must be
        less than dT for horizon_n.

        Args:
            horizon_n: Number of horizon to get marginal likelihoods for.
            i: ilines, array_like.
            x: xlines, array_like.
            t: tlines, array_like.
            relative: Scale likelihoods to [0, 1] interval if True.

        Returns:
            Array of likelihoods.

        """
        if i is None:
            i = range(self.seismic.shape[0])
        else:
            if isinstance(i, int):
                i = range(i, i + 1)
        if x is None:
            x = range(self.seismic.shape[1])
        else:
            if isinstance(x, int):
                x = range(x, x + 1)
        if t is None:
            t = range(self.dTs[horizon_n])
        else:
            if isinstance(x, int):
                t = np.arange(t, t + 1)
            msg = "Only t's between start_t and start_t + dT are valid."
            assert all(t < self.dTs[horizon_n]), msg
        marginal_likelihoods = np.empty((len(i), len(x), len(t)))  # type: ignore
        self.cleaned_seismic += self._get_horizon_seismic(horizon_n)
        for j in i:
            for y in x:  # type: ignore
                for s in t:
                    marginal_likelihoods[
                        j - i[0], y - x[0], s - t[0]  # type: ignore
                    ] = self._marginal_likelihood(horizon_n, j, y, s)
        self.cleaned_seismic -= self._get_horizon_seismic(horizon_n)
        if relative:
            marginal_likelihoods /= marginal_likelihoods.sum(axis=-1, keepdims=True)
        return marginal_likelihoods.squeeze()

    def _get_prior_transition_matrix_depths(
        self,
        horizon_n: int,
        i: int,
        x: int,
        iline_direction: bool = True,
        increasing: bool = True,
    ) -> np.ndarray:
        """Prior probability of transitioning from depth k at x to l at x+1."""
        jumps = scipy.linalg.toeplitz(np.arange(self.dTs[horizon_n]))
        offset = 0.0
        n = 0
        change = 1 if increasing else -1
        if horizon_n != 0:
            n += 1
            if iline_direction:
                above_horizon = self.horizons[horizon_n - 1][i]
                above_horizon_jump = above_horizon[x + change] - above_horizon[x]
            else:
                above_horizon = self.horizons[horizon_n - 1][:, x]
                above_horizon_jump = above_horizon[i + change] - above_horizon[i]
            offset += above_horizon_jump
        if horizon_n != len(self.seed_points_list) - 1:
            n += 1
            if iline_direction:
                below_horizon = self.horizons[horizon_n + 1][i]
                below_horizon_jump = below_horizon[x + change] - below_horizon[x]
            else:
                below_horizon = self.horizons[horizon_n + 1][:, x]
                below_horizon_jump = below_horizon[i + change] - below_horizon[i]
            offset += below_horizon_jump
        offset /= max(1, n)  # mean
        # If the jump in the horizons above and below was high, equally high jumps are
        # not penalized. And we are more certain about the change, so increase df.
        rel_jumps = jumps - offset
        df = self.dfs[horizon_n]
        df *= 10 ** n  # TODO: 10 is magic number, fix
        k = self.ks[horizon_n]
        probs = self._t_dist_core(k * rel_jumps, df)
        # Divison not really necessary, but nice for inspecting prior transition probs
        return probs / probs.sum(axis=0)

    @staticmethod
    def _t_dist_core(x: np.ndarray, df: int) -> np.ndarray:
        """Core of t-distribution, i.e. without normalization constant."""
        return (1 + x ** 2 / df) ** (-(df + 1) / 2)

    def _get_prior_transition_matrix_refl_coeffs(self, horizon_n: int) -> np.ndarray:
        """Prior probability of transitioning from refl coeff k at x to l at x+1."""
        # TODO: Save probs so they don't have to be calculated more than once.
        jumps = scipy.linalg.toeplitz(np.arange(self.dTs[horizon_n]))
        df = self.dfs[horizon_n]
        k = self.ks[horizon_n]
        probs = self._t_dist_core(k * jumps, df)
        return probs

    def _get_horizon_seismic(self, horizon_n: int) -> np.ndarray:
        """Find effects of tracked horizon on seismic.

        ... so that it can be removed when tracking other horizons.

        """
        horizon_seismic = np.zeros_like(self.seismic)
        I, X, _ = self.seismic.shape
        ii, xx = np.mgrid[:I, :X]
        horizon = self.horizons[horizon_n].squeeze()
        horizon_seismic[ii, xx, horizon] = self.reflection_coefficients[horizon_n]
        if self.marginal_likelihood_method == "full":
            wavelet = self.wavelet
        else:
            wavelet = self._cut_wavelet
        horizon_seismic = np.apply_along_axis(
            lambda t: np.convolve(t, wavelet, mode="same"), axis=-1, arr=horizon_seismic
        )
        return horizon_seismic

    def _local_marginal_likelihood(
        self, horizon_n: int, i: int, x: int, t: int, refl_coeff: float = None
    ) -> float:
        """Probability of data at trace (i, x) around t given depth t of the horizon."""
        start_t = self.start_ts[horizon_n]
        seed_points = self.seed_points_list[horizon_n]
        seed_ix = seed_points[:, :2]
        seed_n = np.argwhere(np.all(seed_ix == (i, x), axis=1)).squeeze()
        if seed_n.size > 0:
            # Likelihood zero at all other points than seed point
            if refl_coeff is None:
                return 1 if t == (seed_points[seed_n, 2] - start_t) else 0
            elif self.reflection_coefficient_seeds[horizon_n] is not None:
                jump = self._min_refl_coeff_jump
                seed_refl_coeff = self.reflection_coefficient_seeds[horizon_n][seed_n]
                seed_refl_coeff = np.round(seed_refl_coeff / jump) * jump
                return 1 if refl_coeff == seed_refl_coeff else 0
        data = np.zeros(self._cut_wavelet.size)  # type: ignore
        half_length = self._cut_wavelet.size // 2  # type: ignore
        start = t + self.start_ts[horizon_n] - half_length
        end = t + self.start_ts[horizon_n] + half_length + 1
        T = self.seismic.shape[2]
        seismic = self.cleaned_seismic[i]
        if start < 0:
            data[-start:] = seismic[x, :end]
        elif end > T:
            data[: (T - end)] = seismic[x, start:]
        else:
            data = seismic[x, start:end]
        likelihood = self._get_pdf_local(horizon_n, i, x, refl_coeff)(  # type: ignore
            data
        )
        return self._contain_horizon(horizon_n, i, x, t, likelihood)

    def _get_pdf_local(
        self, horizon_n: int, i: int, x: int, refl_coeff: float
    ) -> Callable[[np.ndarray], float]:
        """Get pdf for local marginal likelihood method."""
        dT = self._cut_wavelet.size  # type: ignore
        t = dT // 2
        W = self._convolution_matrices[horizon_n]
        wavelet = self._cut_wavelet
        pdf = self._get_pdf(horizon_n, i, x, t, dT, wavelet, W, refl_coeff)
        return pdf

    def _get_pdf(
        self,
        horizon_n: int,
        i: int,
        x: int,
        t: int,
        dT: int,
        wavelet: np.ndarray,
        W: np.ndarray,
        refl_coeff: float,
    ) -> Callable[[Any], Any]:
        """Get pdf coherent to wavelet and horizon height t."""
        sigma = self.sigmas[horizon_n]
        r = np.zeros(dT)
        if refl_coeff is not None:
            r[t] = refl_coeff
        else:
            r[t] = self.reflection_coefficients[horizon_n][i, x]
        mean = W @ r
        rel_sigma = sigma / np.abs(wavelet).max()
        cov = rel_sigma ** 2 * W @ W.T
        cov[np.arange(dT), np.arange(dT)] += (rel_sigma * 0.1) ** 2  # White noise
        if self.increasing_var and refl_coeff is not None:
            dist = np.abs(np.arange(dT) - t)
            # Increase variance by rel_sigma * distance to horizon
            cov[np.arange(dT), np.arange(dT)] *= 1 + dist * rel_sigma
        return scipy.stats.multivariate_normal(mean=mean, cov=cov).pdf

    def _contain_horizon(
        self, horizon_n: int, i: int, x: int, t: int, likelihood: float
    ) -> float:
        """Set likelihood to 0 if above/below higher/lower horizons."""
        if horizon_n != 0:
            lower_depth_limit = self.horizons[horizon_n - 1][i, x]
            lower_depth_limit -= self.start_ts[horizon_n]
            lower_depth_limit = max(0, lower_depth_limit)
            if t < lower_depth_limit:
                return 0
        if horizon_n != len(self.seed_points_list) - 1:
            upper_depth_limit = self.horizons[horizon_n + 1][i, x]
            upper_depth_limit -= self.start_ts[horizon_n]
            upper_depth_limit = min(self.dTs[horizon_n] - 1, upper_depth_limit)
            if t > upper_depth_limit:
                return 0
        return likelihood

    def _full_marginal_likelihood(
        self, horizon_n: int, i: int, x: int, t: int, refl_coeff: float = None
    ) -> float:
        """Probability of data at trace (i, x) given the depth t of the horizon."""
        start_t = self.start_ts[horizon_n]
        seed_points = self.seed_points_list[horizon_n]
        seed_ix = seed_points[:, :2]
        seed_n = np.argwhere(np.all(seed_ix == (i, x), axis=1)).squeeze()
        if seed_n.size > 0:
            # Likelihood zero at all other points than seed point
            if refl_coeff is None:
                return 1 if t == (seed_points[seed_n, 2] - start_t) else 0
            else:
                jump = self._min_refl_coeff_jump
                seed_refl_coeff = self.reflection_coefficient_seeds[horizon_n][seed_n]
                seed_refl_coeff = np.round(seed_refl_coeff / jump) * jump
                return 1 if refl_coeff == seed_refl_coeff else 0
        dT = self.dTs[horizon_n]
        wavelet = self.wavelet
        W = self._convolution_matrices[horizon_n]
        pdf = self._get_pdf(
            horizon_n, i, x, t, dT, wavelet, W, refl_coeff  # type: ignore
        )
        seismic = self.cleaned_seismic[i]
        return pdf(seismic[x, start_t : (start_t + dT)])
