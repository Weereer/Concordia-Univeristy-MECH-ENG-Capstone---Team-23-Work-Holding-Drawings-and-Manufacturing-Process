from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np


@dataclass(frozen=True)
class LinearRegionCandidate:
    slope: float
    intercept: float
    r2: float
    i0: int
    i1: int
    window_points: int
    strain_span: float
    stress_span: float
    slope_cv: float

    @property
    def zero_stress_strain(self) -> float:
        if not np.isfinite(self.slope) or self.slope == 0:
            return float("inf")
        return float(-self.intercept / self.slope)

    def as_result(self, strain: np.ndarray, stress: np.ndarray):
        return (
            float(self.slope),
            self.i0,
            self.i1,
            float(self.r2),
            strain,
            stress,
            float(self.intercept),
        )


@dataclass(frozen=True)
class LinearRegionSearchResult:
    strain: np.ndarray
    stress: np.ndarray
    candidates: tuple[LinearRegionCandidate, ...]


def _build_prefix_sum(values: np.ndarray) -> np.ndarray:
    prefix = np.empty(len(values) + 1, dtype=float)
    prefix[0] = 0.0
    np.cumsum(values, dtype=float, out=prefix[1:])
    return prefix


@dataclass(frozen=True)
class _WindowRegressionCache:
    strain: np.ndarray
    stress: np.ndarray
    prefix_x: np.ndarray
    prefix_y: np.ndarray
    prefix_xx: np.ndarray
    prefix_xy: np.ndarray
    prefix_yy: np.ndarray

    @classmethod
    def from_arrays(cls, strain: np.ndarray, stress: np.ndarray):
        x = np.asarray(strain, dtype=float)
        y = np.asarray(stress, dtype=float)
        return cls(
            strain=x,
            stress=y,
            prefix_x=_build_prefix_sum(x),
            prefix_y=_build_prefix_sum(y),
            prefix_xx=_build_prefix_sum(x * x),
            prefix_xy=_build_prefix_sum(x * y),
            prefix_yy=_build_prefix_sum(y * y),
        )

    @staticmethod
    def _window_sum(prefix: np.ndarray, i0: int, i1: int) -> float:
        return float(prefix[i1 + 1] - prefix[i0])

    def fit(self, i0: int, i1: int):
        n = i1 - i0 + 1
        if n < 2:
            return None

        sum_x = self._window_sum(self.prefix_x, i0, i1)
        sum_y = self._window_sum(self.prefix_y, i0, i1)
        sum_xx = self._window_sum(self.prefix_xx, i0, i1)
        sum_xy = self._window_sum(self.prefix_xy, i0, i1)
        sum_yy = self._window_sum(self.prefix_yy, i0, i1)

        sxx = sum_xx - (sum_x * sum_x) / n
        if not np.isfinite(sxx) or sxx <= 0.0:
            return None

        sxy = sum_xy - (sum_x * sum_y) / n
        slope = sxy / sxx
        intercept = (sum_y - slope * sum_x) / n

        if not np.isfinite(slope) or not np.isfinite(intercept):
            return None

        syy = sum_yy - (sum_y * sum_y) / n
        if not np.isfinite(syy):
            return None

        if syy <= 0.0:
            r2 = 0.0
        else:
            sse = syy - slope * sxy
            if sse < 0.0 and np.isclose(sse, 0.0, atol=max(abs(syy), 1.0) * 1e-12):
                sse = 0.0
            r2 = 1.0 - sse / syy
            if r2 < 0.0 and np.isclose(r2, 0.0, atol=1e-12):
                r2 = 0.0
            elif r2 > 1.0 and np.isclose(r2, 1.0, atol=1e-12):
                r2 = 1.0

        if not np.isfinite(r2):
            return None

        return float(slope), float(intercept), float(r2)


@lru_cache(maxsize=None)
def _segment_edges(window_points: int) -> tuple[tuple[int, int], ...]:
    edges = np.linspace(0, window_points, 4, dtype=int)
    return tuple((int(start), int(end)) for start, end in zip(edges[:-1], edges[1:]))


def _compute_slope_cv(cache: _WindowRegressionCache, i0: int, i1: int) -> float:
    window_points = i1 - i0 + 1
    if window_points < 6:
        return float("inf")

    local_slopes = []

    for start, end in _segment_edges(window_points):
        seg_i0 = i0 + start
        seg_i1 = i0 + end - 1
        if seg_i1 - seg_i0 + 1 < 2:
            return float("inf")

        fit = cache.fit(seg_i0, seg_i1)
        if fit is None:
            return float("inf")

        local_slopes.append(abs(fit[0]))

    slopes = np.asarray(local_slopes, dtype=float)
    mean_slope = np.mean(slopes)

    if not np.isfinite(mean_slope) or mean_slope <= 0:
        return float("inf")

    return float(np.std(slopes) / mean_slope)


def _build_candidate(
    cache: _WindowRegressionCache,
    i0: int,
    i1: int,
):
    fit = cache.fit(i0, i1)
    if fit is None:
        return None

    slope, intercept, r2 = fit

    return LinearRegionCandidate(
        slope=slope,
        intercept=intercept,
        r2=r2,
        i0=i0,
        i1=i1,
        window_points=i1 - i0 + 1,
        strain_span=float(abs(cache.strain[i1] - cache.strain[i0])),
        stress_span=float(abs(cache.stress[i1] - cache.stress[i0])),
        slope_cv=_compute_slope_cv(cache, i0, i1),
    )


def _clean_curve(strain, stress, *, threshold_stress=0.0):
    s = np.asarray(strain, dtype=float)
    t = np.asarray(stress, dtype=float)

    mask = np.isfinite(s) & np.isfinite(t) & (t > threshold_stress)
    return s[mask], t[mask]


def enumerate_linear_region_candidates(
    strain,
    stress,
    *,
    window_min=20,
    window_max=None,
    window_step=5,
    threshold_stress=0.0,
    start_idx=0,
    end_idx=None,
):
    s_clean, t_clean = _clean_curve(strain, stress, threshold_stress=threshold_stress)

    if len(s_clean) < 2:
        return LinearRegionSearchResult(
            strain=s_clean,
            stress=t_clean,
            candidates=tuple(),
        )

    start = max(int(start_idx), 0)
    stop = len(s_clean) - 1 if end_idx is None else min(int(end_idx), len(s_clean) - 1)

    if stop - start + 1 < 2:
        return LinearRegionSearchResult(
            strain=s_clean,
            stress=t_clean,
            candidates=tuple(),
        )

    lower = max(int(window_min), 2)
    upper = stop - start + 1 if window_max is None else min(int(window_max), stop - start + 1)
    step = max(int(window_step), 1)

    if upper < lower:
        return LinearRegionSearchResult(
            strain=s_clean,
            stress=t_clean,
            candidates=tuple(),
        )

    candidates = []
    regression_cache = _WindowRegressionCache.from_arrays(s_clean, t_clean)

    for window_points in range(lower, upper + 1, step):
        last_start = stop - window_points + 1
        for i0 in range(start, last_start + 1):
            i1 = i0 + window_points - 1
            candidate = _build_candidate(regression_cache, i0, i1)
            if candidate is not None:
                candidates.append(candidate)

    return LinearRegionSearchResult(
        strain=s_clean,
        stress=t_clean,
        candidates=tuple(candidates),
    )


def _candidate_tier(candidate: LinearRegionCandidate, r2_min: float) -> int:
    if not np.isfinite(candidate.slope) or candidate.slope <= 0:
        return 3

    if candidate.r2 >= r2_min:
        return 1

    return 2


def _eligible_sort_key(candidate: LinearRegionCandidate, r2_min: float):
    tier_score = 2 if _candidate_tier(candidate, r2_min) == 1 else 1
    return (
        tier_score,
        candidate.window_points,
        -candidate.slope_cv,
        candidate.strain_span,
        candidate.r2,
        -candidate.i0,
    )


def _numeric_sort_key(candidate: LinearRegionCandidate):
    return (candidate.r2, candidate.window_points, -candidate.i0)


def find_linear_region_properties(
    strain,
    stress,
    *,
    window_min=200,
    window_step=100,
    r2_min=0.95,
    threshold_stress=1.24e6,
    fallback_min_points=20,
):
    _ = fallback_min_points

    search = enumerate_linear_region_candidates(
        strain,
        stress,
        window_min=window_min,
        window_step=window_step,
        threshold_stress=threshold_stress,
    )

    if not search.candidates:
        return np.nan, None, None, None, search.strain, search.stress, np.nan

    best_eligible = None
    best_numeric = None

    for candidate in search.candidates:
        if best_numeric is None or _numeric_sort_key(candidate) > _numeric_sort_key(best_numeric):
            best_numeric = candidate

        tier = _candidate_tier(candidate, r2_min)
        if tier in (1, 2):
            if best_eligible is None or _eligible_sort_key(candidate, r2_min) > _eligible_sort_key(best_eligible, r2_min):
                best_eligible = candidate

    selected = best_eligible if best_eligible is not None else best_numeric
    if selected is None:
        return np.nan, None, None, None, search.strain, search.stress, np.nan

    return selected.as_result(search.strain, search.stress)
