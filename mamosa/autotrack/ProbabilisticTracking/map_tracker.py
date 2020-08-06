from mamosa.autotrack.ProbabilisticTracking import ProbabilisticTracker
import numpy as np
from typing import Tuple, List
import warnings

# TODO: Vectorize where possible.


class MAPTracker(ProbabilisticTracker):
    """This class is used to find horizon MAP approximations, one inline at a time."""

    # TODO: Document self.map_probabilities attribute.

    def __init__(
        self,
        seismic: np.ndarray,
        wavelet: np.ndarray,
        marginal_likelihood_method: str = "local",
        increasing_var: bool = False,
    ) -> None:
        super().__init__(seismic, wavelet, marginal_likelihood_method, increasing_var)
        self.map_probabilities: List[float] = []

    def _track(self, horizon_n: int) -> np.ndarray:
        """Track the horizon using a Viterbi based method."""
        # TODO: Link to report
        if len(self.seed_points_list[horizon_n]) > 0:
            i_initial = self.seed_points_list[horizon_n][0, 0]
        else:
            i_initial = 0
        I, X, _ = self.seismic.shape
        self.horizons.insert(horizon_n, np.empty((I, X), dtype=int))
        start_t = self.start_ts[horizon_n]
        self.horizons[horizon_n][i_initial] = (
            self._run_viterbi(horizon_n, i_initial) + start_t
        )
        reflection_coefficient_levels = self._run_viterbi(horizon_n, i_initial, True)
        reflection_coefficients = self.reflection_coefficients[horizon_n]
        reflection_coefficients[i_initial] = self._coeffs(
            horizon_n, i_initial, reflection_coefficient_levels
        )
        for i in range(i_initial - 1, -1, -1):
            reflection_coefficients[i] = reflection_coefficients[i + 1]
            self.horizons[horizon_n][i] = self._run_viterbi(horizon_n, i) + start_t
            reflection_coefficient_levels = self._run_viterbi(horizon_n, i, True)
            reflection_coefficients[i] = self._coeffs(
                horizon_n, i, reflection_coefficient_levels
            )
        for i in range(i_initial + 1, I):
            reflection_coefficients[i] = reflection_coefficients[i - 1]
            self.horizons[horizon_n][i] = self._run_viterbi(horizon_n, i) + start_t
            reflection_coefficient_levels = self._run_viterbi(horizon_n, i, True)
            reflection_coefficients[i] = self._coeffs(
                horizon_n, i, reflection_coefficient_levels
            )
        return self.horizons[horizon_n]

    def _coeffs(
        self, horizon_n: int, i: int, reflection_levels: np.ndarray
    ) -> np.ndarray:
        """Convert reflection coefficient levels to actual coefficients."""
        coeffs = self._temp_refl_coeff_domain[reflection_levels]  # type: ignore
        seed_indices = np.argwhere(self.seed_points_list[horizon_n][:, 0] == i)
        seed_x = self.seed_points_list[horizon_n][seed_indices, 1]
        coeffs[seed_x] = self.reflection_coefficients[horizon_n][i, seed_x]
        return coeffs

    def _run_viterbi(
        self, horizon_n: int, i: int, refl_coeffs: bool = False
    ) -> np.ndarray:
        """Track horizon in iline i using the Viterbi algorithm. TODO: Ref to report."""
        if refl_coeffs:
            self._set_temp_refl_coeff_domain(horizon_n, i)
        X = self.seismic.shape[1]
        dT = self._get_dT(horizon_n, refl_coeffs)
        # Point (x,t) of array is the depth at x-1 for the most probable path to (x,t)
        from_array = np.full((X, dT), fill_value=-1)
        x1_probabilities, transition_matrices = self._get_transition_matrices(
            horizon_n, i, refl_coeffs
        )
        # Always dT routes considered
        route_probabilities = x1_probabilities
        r = np.arange(dT)
        for x in range(X - 1):
            all_probabilities = (
                route_probabilities.reshape(-1, 1) * transition_matrices[x]
            )
            from_array[x] = all_probabilities.argmax(axis=0)
            route_probabilities = all_probabilities[from_array[x], r]
        end_point = route_probabilities.argmax()
        route = self._backtrack_route(from_array, end_point)
        self.map_probabilities.insert(horizon_n, route_probabilities[end_point])
        return route

    def _set_temp_refl_coeff_domain(self, horizon_n: int, i: int):
        """Set reflection coefficient domain for inline i."""
        refl_coeffs = self.reflection_coefficients[horizon_n][i]
        jump = self._min_refl_coeff_jump
        new = self._refl_n_new / 2
        min = np.round(refl_coeffs.min() / jump) - new
        max = np.round(refl_coeffs.max() / jump) + new
        self._temp_refl_coeff_domain = np.arange(min, max + 1) * jump

    def _get_dT(self, horizon_n: int, refl_coeffs: bool) -> int:
        """Return dT, depending on whether depths or coefficients are considered"""
        if refl_coeffs:
            return self._temp_refl_coeff_domain.size  # type: ignore
        return self.dTs[horizon_n]

    def _get_transition_matrices(
        self, horizon_n: int, i: int, refl_coeffs: bool
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Recursively calculate transition probabilities. Se the report for details."""
        # TODO: Ref to report.
        X = self.seismic.shape[1]
        dT = self._get_dT(horizon_n, refl_coeffs)
        transition_matrices = np.empty((X - 1, dT, dT))
        sums = np.ones(dT)
        v = self._get_v(horizon_n, i, X - 1, sums, refl_coeffs)
        transition_matrices, sums = self._get_transition_probabilities_step(
            horizon_n, i, X - 2, v, transition_matrices, refl_coeffs
        )
        for x in range(X - 3, -1, -1):
            v = self._get_v(horizon_n, i, x + 1, sums, refl_coeffs)
            transition_matrices, sums = self._get_transition_probabilities_step(
                horizon_n, i, x, v, transition_matrices, refl_coeffs
            )
        v = self._get_v(horizon_n, i, 0, sums, refl_coeffs)
        x1_probabilities = v / v.sum()
        return x1_probabilities, transition_matrices

    def _get_v(
        self, horizon_n: int, i: int, x: int, sums: np.ndarray, refl_coeffs: bool
    ) -> np.ndarray:
        """Get vector of v_k for all s_k (see the report for details)."""
        # TODO: Ref to report.
        dT = self._get_dT(horizon_n, refl_coeffs)
        if len(self.seed_points_list[horizon_n]) > 0:
            i_initial = self.seed_points_list[horizon_n][0, 0]
        else:
            i_initial = 0
        if i == i_initial:
            prior_transition_probs_x = np.ones(dT)
        else:
            increasing_i = i > i_initial
            change = 1 if increasing_i else -1
            if refl_coeffs:
                prev_i_refl_level = self._get_refl_coeff_level(horizon_n, i, x)
                prior = self._get_prior_transition_matrix_refl_coeffs
                prior_transition_probs_x = prior(horizon_n)[prev_i_refl_level]
            else:
                prev_i_depth = self.horizons[horizon_n][i - change, x]
                prev_i_depth -= self.start_ts[horizon_n]
                prior_transition_probs_x = self._get_prior_transition_matrix_depths(
                    horizon_n, i - 1, x, False, increasing_i
                )[prev_i_depth]
        if refl_coeffs:
            t = self.horizons[horizon_n][i, x] - self.start_ts[horizon_n]
            coeffs = self._temp_refl_coeff_domain
            v = np.array(
                [
                    self._marginal_likelihood(
                        horizon_n, i, x, t, coeffs[refl_t]  # type: ignore
                    )
                    * prior_transition_probs_x[refl_t]
                    * sums[refl_t]
                    for refl_t in range(dT)
                ]
            )
        else:
            v = np.array(
                [
                    self._marginal_likelihood(horizon_n, i, x, t)
                    * prior_transition_probs_x[t]
                    * sums[t]
                    for t in range(dT)
                ]
            )
        # Avoid vanishing/exploding vs, see report for details
        v /= v.mean()
        return v

    def _get_refl_coeff_level(self, horizon_n: int, i: int, x: int) -> np.ndarray:
        """Get level of reflection coefficient in domain."""
        refl_coeff = self.reflection_coefficients[horizon_n][i, x]
        jump = self._min_refl_coeff_jump
        refl_coeff = np.round(refl_coeff / jump) * jump
        return np.argwhere(self._temp_refl_coeff_domain == refl_coeff).squeeze()

    def _get_transition_probabilities_step(
        self,
        horizon_n: int,
        i: int,
        x: int,
        v: np.ndarray,
        transition_matrices: np.ndarray,
        refl_coeffs: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Set transition probabilities from xline x. Se the report for details."""
        # TODO: Catch zeros in sums. Limit df and k in parent method track_horizon?
        dT = self._get_dT(horizon_n, refl_coeffs)
        sums = np.zeros(dT)
        if refl_coeffs:
            prior_transition_matrix = self._get_prior_transition_matrix_refl_coeffs(
                horizon_n
            )
        else:
            prior_transition_matrix = self._get_prior_transition_matrix_depths(
                horizon_n, i, x
            )
        for s in range(dT):
            for t in range(dT):
                sums[s] += prior_transition_matrix[s, t] * v[t]
        if np.any(sums == 0):
            warnings.warn("Zeros encountered. Try reducing df or k.")
        for s in range(dT):
            for t in range(dT):
                transition_matrices[x, s, t] = (
                    prior_transition_matrix[s, t] * v[t] / sums[s]
                )
        return transition_matrices, sums

    def _backtrack_route(self, from_array: np.ndarray, end_point: int) -> np.ndarray:
        """Track route back from the most probable endpoint using from_array."""
        X = self.seismic.shape[1]
        route = np.empty(X, dtype=int)
        route[-1] = end_point
        for x in range(X - 1):
            route[X - x - 2] = from_array[X - x - 2, route[X - x - 1]]
        return route

    def a_posteriori_probability(
        self, route: np.ndarray, i: int, horizon_n: int = 0
    ) -> float:
        """Return a posteriori probability of route (depths) at iline i.

        Args:
            route: ndarray of shape (X,); horizon route (depths).
            i: Inline number.
            horizon_n: Horizon number.

        Returns:
            A posteriori route probability.

        """
        err_msg = (
            f"Horizon {horizon_n} not tracked yet; only "
            f"{len(self.seed_points_list)} horizons tracked."
        )
        assert horizon_n < len(self.seed_points_list), err_msg
        X = self.seismic.shape[1]
        r = route.squeeze() - self.start_ts[horizon_n]
        assert r.shape == (X,), "Route length must be seismic.shape[1]."
        self.cleaned_seismic += self._get_horizon_seismic(horizon_n)
        x1_probabilities, transition_matrices = self._get_transition_matrices(
            horizon_n, i, False
        )
        self.cleaned_seismic -= self._get_horizon_seismic(horizon_n)
        probability = x1_probabilities[r[0]]
        for x in range(X - 1):
            probability *= transition_matrices[x, r[x], r[x + 1]]
        return probability

    def get_marginal_posteriors(self, i: int, horizon_n: int = 0) -> np.ndarray:
        """Returns the array of marginal posterior probabilities for iline i.

        Args:
            i: Inline number.
            horizon_n: Horizon number.

        Returns:
            ndarray of shape (X, T); marginal probabilities for the possible depths at
            inline i.

        """
        err_msg = (
            f"Horizon {horizon_n} not tracked yet; only "
            f"{len(self.seed_points_list)} horizons tracked."
        )
        assert horizon_n < len(self.seed_points_list), err_msg
        X = self.seismic.shape[1]
        dT = self.dTs[horizon_n]
        self.cleaned_seismic += self._get_horizon_seismic(horizon_n)
        marginal_probs = np.empty((X, dT))
        marginal_probs[0, :], transition_matrices = self._get_transition_matrices(
            horizon_n, i, False
        )
        self.cleaned_seismic -= self._get_horizon_seismic(horizon_n)
        for x in range(1, X):
            for t in range(dT):
                marginal_probs[x, t] = (
                    transition_matrices[x - 1, :, t] * marginal_probs[x - 1]
                ).sum()
        return marginal_probs
