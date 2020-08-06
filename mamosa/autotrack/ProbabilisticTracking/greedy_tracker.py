from mamosa.autotrack.ProbabilisticTracking import ProbabilisticTracker
import numpy as np
from typing import Tuple, Any, List

# TODO: Use scipy.sparse for frontier


class GreedyTracker(ProbabilisticTracker):
    """Greedy probability based horizon tracker.

    TODO: Ref to report.

    """

    def __init__(
        self,
        seismic: np.ndarray,
        wavelet: np.ndarray,
        marginal_likelihood_method: str = "local",
        increasing_var: bool = False,
    ) -> None:
        super().__init__(seismic, wavelet, marginal_likelihood_method, increasing_var)
        self.max_jumps: List[int] = []

    def _track(self, horizon_n: int, max_jump: int = None) -> np.ndarray:
        """Abstract method; track the horizon using a greedy method."""
        # TODO: Ref to report.
        msg = "Greedy tracker needs at least one seed point."
        assert self.seed_points_list[horizon_n].size > 0, msg
        I, X, _ = self.seismic.shape
        self.horizons.insert(horizon_n, np.full((I, X), -1))
        if max_jump is None:
            self.max_jumps.insert(horizon_n, self.dTs[horizon_n])
        else:
            self.max_jumps.insert(horizon_n, max_jump)
        frontier, score = self._initialize_frontier(horizon_n)
        horizon = self.horizons[horizon_n]
        while np.any(frontier):
            i, x, t = np.unravel_index(score.argmax(), score.shape)
            frontier[i, x] = 0
            score[i, x] = 0
            horizon[i, x] = t
            self._estimate_reflection_coefficient(horizon_n, i, x)
            if i < I - 1 and horizon[i + 1, x] == -1:
                frontier[i + 1, x], score[i + 1, x] = self._add_to_frontier(
                    horizon_n, i, x, False, True, frontier[i + 1, x], score[i + 1, x]
                )
            if i > 0 and horizon[i - 1, x] == -1:
                frontier[i - 1, x], score[i - 1, x] = self._add_to_frontier(
                    horizon_n, i, x, False, False, frontier[i - 1, x], score[i - 1, x]
                )
            if x < X - 1 and horizon[i, x + 1] == -1:
                frontier[i, x + 1], score[i, x + 1] = self._add_to_frontier(
                    horizon_n, i, x, True, True, frontier[i, x + 1], score[i, x + 1]
                )
            if x > 0 and horizon[i, x - 1] == -1:
                frontier[i, x - 1], score[i, x - 1] = self._add_to_frontier(
                    horizon_n, i, x, True, False, frontier[i, x - 1], score[i, x - 1]
                )
        horizon += self.start_ts[horizon_n]
        return horizon

    def _estimate_reflection_coefficient(self, horizon_n: int, i: int, x: int) -> None:
        """Estimate reflection coefficient in a column by maximizing likelihood."""
        old_estimate = self.reflection_coefficients[horizon_n][i, x]
        min_jump = self._min_refl_coeff_jump
        n_new = self._refl_n_new
        low = old_estimate - min_jump * n_new
        high = old_estimate + min_jump * n_new
        refl_range = np.arange(low, high + min_jump, min_jump)
        new_estimate = refl_range[0]
        t = self.horizons[horizon_n][i, x]
        max_likelihood = self._marginal_likelihood(horizon_n, i, x, t, new_estimate)
        for refl_coeff in refl_range[1:]:
            likelihood = self._marginal_likelihood(horizon_n, i, x, t, refl_coeff)
            if likelihood > max_likelihood:
                max_likelihood = likelihood
                new_estimate = refl_coeff
        self.reflection_coefficients[horizon_n][i, x] = new_estimate

    def _initialize_frontier(self, horizon_n: int) -> Tuple[np.ndarray, np.ndarray]:
        """Initialize frontier indicator array and score array."""
        seed_points = self.seed_points_list[horizon_n]
        I, X, _ = self.seismic.shape
        dT = self.dTs[horizon_n]
        start_t = self.start_ts[horizon_n]
        frontier = np.zeros((I, X, dT))
        score = np.zeros((I, X, dT))
        for seed in seed_points:
            i, x, t = seed
            self.horizons[horizon_n][i, x] = t - start_t
            seed_ix = seed_points[:, :2]
            if i < I - 1 and not self._is_in_array((i + 1, x), seed_ix):
                frontier[i + 1, x], score[i + 1, x] = self._add_to_frontier(
                    horizon_n, i, x, False, True, frontier[i + 1, x], score[i + 1, x]
                )
            if i > 0 and not self._is_in_array((i - 1, x), seed_ix):
                frontier[i - 1, x], score[i - 1, x] = self._add_to_frontier(
                    horizon_n, i, x, False, False, frontier[i - 1, x], score[i - 1, x]
                )
            if x < X - 1 and not self._is_in_array((i, x + 1), seed_ix):
                frontier[i, x + 1], score[i, x + 1] = self._add_to_frontier(
                    horizon_n, i, x, True, True, frontier[i, x + 1], score[i, x + 1]
                )
            if x > 0 and not self._is_in_array((i, x - 1), seed_ix):
                frontier[i, x - 1], score[i, x - 1] = self._add_to_frontier(
                    horizon_n, i, x, True, False, frontier[i, x - 1], score[i, x - 1]
                )
        return frontier, score

    @staticmethod
    def _is_in_array(row: Any, arr: np.ndarray) -> bool:
        """Return True if row is in array."""
        return (row == arr).all(axis=1).any()

    def _add_to_frontier(
        self,
        horizon_n: int,
        i: int,
        x: int,
        iline_direction: bool,
        increasing: bool,
        frontier: np.ndarray,
        scores: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Add column and expansion probabilities to frontier."""
        _, _, low, high, _, _ = self._get_data(
            horizon_n, i, x, iline_direction, increasing
        )
        new_scores = self._get_scores(horizon_n, i, x, iline_direction, increasing)
        in_frontier = frontier[low : high + 1] > 0
        if in_frontier.any():
            # Scale scores to make them comparable. TODO: Ref to report.
            exp1 = frontier[in_frontier]
            exp2 = 1 / (frontier[in_frontier] + 1)
            scores[low : high + 1][in_frontier] = (
                (scores[low : high + 1][in_frontier] ** exp1) * new_scores[in_frontier]
            ) ** exp2
        scores[low : high + 1][~in_frontier] = new_scores[~in_frontier]
        frontier[low : high + 1] += 1
        return frontier, scores

    def _get_scores(
        self, horizon_n: int, i: int, x: int, iline_direction: bool, increasing: bool
    ) -> np.ndarray:
        """Get scores for depths to add to frontier at new column."""
        # TODO: Ref to report.
        dT, t, low, high, i_new, x_new = self._get_data(
            horizon_n, i, x, iline_direction, increasing
        )
        # Use neighbor reflection coefficient as estimate
        self.reflection_coefficients[horizon_n][
            i_new, x_new
        ] = self.reflection_coefficients[horizon_n][i, x]
        scores = np.empty(high - low + 1)
        phi = self._get_prior_transition_matrix_depths(
            horizon_n, i, x, iline_direction, increasing
        )[t, low : high + 1]
        for s in range(low, high + 1):
            scores[s - low] = phi[s - low] * self._marginal_likelihood(
                horizon_n, i_new, x_new, s
            )
        return scores

    def _get_data(
        self, horizon_n: int, i: int, x: int, iline_direction: bool, increasing: bool
    ) -> Tuple[int, int, int, int, int, int]:
        dT = self.dTs[horizon_n]
        t = self.horizons[horizon_n][i, x] - self.start_ts[horizon_n]
        max_jump = self.max_jumps[horizon_n]
        low = max(0, t - max_jump)
        high = min(dT - 1, t + max_jump)
        if iline_direction:
            i_new = i
            x_new = x + (1 if increasing else -1)
        else:
            i_new = i + (1 if increasing else -1)
            x_new = x
        return dT, t, low, high, i_new, x_new
