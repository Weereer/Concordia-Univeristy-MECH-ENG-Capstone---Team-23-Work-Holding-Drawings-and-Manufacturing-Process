from __future__ import annotations

import warnings

import numpy as np

from shared_core.curve_components import clean_curve_data
from shared_core.linear_region import LinearRegionCandidate, enumerate_linear_region_candidates

from .constants import (
    COMPRESSION_CANDIDATE_WINDOW_MAX,
    COMPRESSION_CANDIDATE_WINDOW_MIN,
    COMPRESSION_CANDIDATE_WINDOW_STEP,
    COMPRESSION_GEOMETRIC_REHARDENING_RATIO,
    COMPRESSION_PRIMARY_R2_MIN,
    COMPRESSION_SHORT_RAMP_PRE_YIELD_STRAIN_MAX,
    COMPRESSION_SHORT_RAMP_RELAXED_R2_MIN,
    COMPRESSION_SHORT_RAMP_RELAXED_SLOPE_CV_MAX,
    COMPRESSION_SHORT_RAMP_RELAXED_WINDOW_MAX,
    COMPRESSION_SHORT_RAMP_RELAXED_WINDOW_MIN,
    COMPRESSION_SHORT_RAMP_RELAXED_WINDOW_STEP,
    COMPRESSION_SHORT_RAMP_RELAXED_ZERO_STRAIN_MAX,
    COMPRESSION_SHORT_RAMP_R2_MIN,
    COMPRESSION_SHORT_RAMP_SLOPE_CV_MAX,
    COMPRESSION_SHORT_RAMP_USABLE_POINTS_MAX,
    COMPRESSION_SHORT_RAMP_WINDOW_MAX,
    COMPRESSION_SHORT_RAMP_WINDOW_MIN,
    COMPRESSION_SHORT_RAMP_WINDOW_STEP,
    COMPRESSION_SHORT_RAMP_ZERO_STRAIN_MAX,
    COMPRESSION_SLOPE_CV_MAX,
    COMPRESSION_SOFTENING_CONSECUTIVE,
    COMPRESSION_SOFTENING_DROP_RATIO,
    COMPRESSION_SOFTENING_START_STRESS_RATIO,
    COMPRESSION_SOFTENING_WINDOW,
    COMPRESSION_STARTUP_SKIP_FRACTION,
    COMPRESSION_STARTUP_SKIP_STRAIN_MAX,
    COMPRESSION_WINDUP_CONSECUTIVE,
    COMPRESSION_WINDUP_CV_MAX,
    COMPRESSION_WINDUP_TARGET_RATIO,
    COMPRESSION_WINDUP_WINDOW,
    COMPRESSION_ZERO_STRAIN_MAX,
    COMPRESSION_ZERO_STRAIN_NEAR_ORIGIN_MAX,
    DECISION_ACCEPTED_CLUSTERED,
    DECISION_REJECTED_GEOMETRIC,
    DECISION_REJECTED_NO_FIT,
    FIT_MODE_DIRECT,
    FIT_MODE_NOISE_TRIMMED,
    FIT_MODE_PRE_YIELD_CLIPPED,
    FIT_MODE_SHORT_RAMP_EXCEPTION,
    FIT_MODE_SHORT_RAMP_RELAXED,
)
from .models import CompressionFitCandidate


_RANK_WARNING = getattr(getattr(np, "exceptions", None), "RankWarning", RuntimeWarning)


def _safe_linear_fit(x: np.ndarray, y: np.ndarray):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", _RANK_WARNING)
            slope, intercept = np.polyfit(x, y, 1)
    except Exception:
        return None

    if not np.isfinite(slope) or not np.isfinite(intercept):
        return None

    yhat = slope * x + intercept
    if not np.all(np.isfinite(yhat)):
        return None

    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 0.0 if ss_tot == 0 else 1.0 - ss_res / ss_tot
    if not np.isfinite(r2):
        return None

    return float(slope), float(intercept), float(r2)


def _median_smooth(values: np.ndarray, radius: int = 1) -> np.ndarray:
    if values.size == 0:
        return values

    smoothed = np.empty_like(values, dtype=float)
    for i in range(len(values)):
        lo = max(0, i - radius)
        hi = min(len(values), i + radius + 1)
        smoothed[i] = float(np.median(values[lo:hi]))
    return smoothed


def detect_pre_yield_cutoff(
    strain,
    stress,
    *,
    rolling_window=COMPRESSION_SOFTENING_WINDOW,
    drop_ratio=COMPRESSION_SOFTENING_DROP_RATIO,
    consecutive=COMPRESSION_SOFTENING_CONSECUTIVE,
    rehardening_ratio=COMPRESSION_GEOMETRIC_REHARDENING_RATIO,
    start_stress_ratio=COMPRESSION_SOFTENING_START_STRESS_RATIO,
):
    strain = np.asarray(strain, dtype=float)
    stress = np.asarray(stress, dtype=float)

    n = len(strain)
    if n < max(int(rolling_window), 3) + consecutive:
        return n - 1, False

    window = max(int(rolling_window), 3)
    slopes = []

    for start in range(0, n - window + 1):
        fit = _safe_linear_fit(strain[start : start + window], stress[start : start + window])
        slopes.append(float("nan") if fit is None else float(fit[0]))

    slopes = np.asarray(slopes, dtype=float)
    finite_mask = np.isfinite(slopes)
    if not np.any(finite_mask):
        return n - 1, False

    finite_slopes = slopes[finite_mask]
    floor = 0.10 * np.nanmax(finite_slopes)
    slopes = np.where(np.isfinite(slopes), slopes, floor)
    smoothed = _median_smooth(slopes, radius=1)
    running_median = np.array([np.median(smoothed[: i + 1]) for i in range(len(smoothed))], dtype=float)
    best_prior = np.maximum.accumulate(running_median)

    search_start = max(1, COMPRESSION_CANDIDATE_WINDOW_MIN)
    finite_stress = np.abs(stress[np.isfinite(stress)])
    if finite_stress.size > 0:
        start_stress_floor = float(start_stress_ratio) * float(np.nanmax(finite_stress))
        if np.isfinite(start_stress_floor) and start_stress_floor > 0:
            meaningful_stress_idx = np.flatnonzero(np.abs(stress) >= start_stress_floor)
            if meaningful_stress_idx.size > 0:
                search_start = max(
                    search_start,
                    max(1, int(meaningful_stress_idx[0]) - window // 2),
                )

    for start in range(search_start, len(smoothed) - consecutive + 1):
        prior_best = best_prior[start - 1]
        if not np.isfinite(prior_best) or prior_best <= 0:
            continue

        window_values = smoothed[start : start + consecutive]
        if np.all(window_values < drop_ratio * prior_best):
            cutoff_idx = min(n - 1, start + window // 2)
            post_values = smoothed[start + consecutive :]
            geometric_instability = bool(
                post_values.size > 0
                and np.nanmax(post_values) > rehardening_ratio * prior_best
            )
            return cutoff_idx, geometric_instability

    return n - 1, False


def detect_startup_windup_end(
    strain,
    stress,
    *,
    rolling_window=COMPRESSION_WINDUP_WINDOW,
    target_ratio=COMPRESSION_WINDUP_TARGET_RATIO,
    slope_cv_max=COMPRESSION_WINDUP_CV_MAX,
    consecutive=COMPRESSION_WINDUP_CONSECUTIVE,
    max_idx=None,
):
    strain = np.asarray(strain, dtype=float)
    stress = np.asarray(stress, dtype=float)

    n = len(strain)
    if n < 2:
        return 0

    window = max(int(rolling_window), 3)
    stop = n - 1 if max_idx is None else min(int(max_idx), n - 1)
    if stop + 1 < window + consecutive:
        return 0

    slopes = []
    for start in range(0, stop - window + 2):
        fit = _safe_linear_fit(strain[start : start + window], stress[start : start + window])
        slope = float("nan") if fit is None else float(fit[0])
        slopes.append(slope if slope > 0 else float("nan"))

    slopes = np.asarray(slopes, dtype=float)
    finite_slopes = slopes[np.isfinite(slopes)]
    if finite_slopes.size < consecutive:
        return 0

    fill = float(np.nanmedian(finite_slopes))
    smoothed = _median_smooth(np.where(np.isfinite(slopes), slopes, fill), radius=1)
    target_slope = float(np.nanpercentile(smoothed, 80))
    if not np.isfinite(target_slope) or target_slope <= 0:
        return 0

    slope_floor = target_ratio * target_slope
    for start in range(0, len(smoothed) - consecutive + 1):
        window_values = smoothed[start : start + consecutive]
        mean_slope = float(np.mean(window_values))
        if mean_slope <= 0:
            continue
        slope_cv = float(np.std(window_values) / mean_slope)
        if np.all(window_values >= slope_floor) and slope_cv <= slope_cv_max:
            return min(stop, start + window // 2)

    return 0


def _base_fit_mode(candidate: LinearRegionCandidate, strain: np.ndarray, yield_cutoff_idx: int | None) -> str:
    zero_strain = candidate.zero_stress_strain
    if candidate.i0 > 0 and strain[candidate.i0] > 0:
        if (
            yield_cutoff_idx is not None
            and yield_cutoff_idx < len(strain) - 1
            and np.isfinite(zero_strain)
            and abs(zero_strain) <= COMPRESSION_ZERO_STRAIN_NEAR_ORIGIN_MAX
        ):
            return FIT_MODE_PRE_YIELD_CLIPPED
        return FIT_MODE_NOISE_TRIMMED
    if yield_cutoff_idx is not None and yield_cutoff_idx < len(strain) - 1:
        return FIT_MODE_PRE_YIELD_CLIPPED
    return FIT_MODE_DIRECT


def _candidate_sort_key(candidate: CompressionFitCandidate):
    return (
        -candidate.window_points,
        abs(candidate.zero_strain),
        candidate.slope_cv,
        -candidate.r2,
        candidate.start_strain,
    )


def _startup_skip_zero_strain_max(
    strain: np.ndarray,
    yield_cutoff_idx: int | None,
    *,
    zero_strain_max: float,
) -> float:
    if strain.size == 0:
        return float(zero_strain_max)

    stop = len(strain) - 1 if yield_cutoff_idx is None else min(max(int(yield_cutoff_idx), 0), len(strain) - 1)
    pre_yield_origin = min(0.0, float(np.nanmin(strain[: stop + 1])))
    pre_yield_span = max(0.0, float(strain[stop] - pre_yield_origin))
    return max(
        float(zero_strain_max),
        min(
            COMPRESSION_STARTUP_SKIP_STRAIN_MAX,
            COMPRESSION_STARTUP_SKIP_FRACTION * pre_yield_span,
        ),
    )


def _convert_candidate(
    candidate: LinearRegionCandidate,
    strain: np.ndarray,
    *,
    yield_cutoff_idx: int | None,
    r2_min: float,
    slope_cv_max: float,
    zero_strain_max: float,
    fit_mode_override: str | None = None,
):
    zero_strain = candidate.zero_stress_strain
    fit_mode = fit_mode_override or _base_fit_mode(candidate, strain, yield_cutoff_idx)
    positive_zero_strain_max = float(zero_strain_max)
    if fit_mode == FIT_MODE_NOISE_TRIMMED:
        positive_zero_strain_max = _startup_skip_zero_strain_max(
            strain,
            yield_cutoff_idx,
            zero_strain_max=zero_strain_max,
        )

    if not np.isfinite(candidate.slope) or candidate.slope <= 0:
        return None
    if not np.isfinite(candidate.r2) or candidate.r2 < r2_min:
        return None
    if not np.isfinite(candidate.slope_cv) or candidate.slope_cv > slope_cv_max:
        return None
    if (
        not np.isfinite(zero_strain)
        or zero_strain < -float(zero_strain_max)
        or zero_strain > positive_zero_strain_max
    ):
        return None

    return CompressionFitCandidate(
        slope=float(candidate.slope),
        intercept=float(candidate.intercept),
        r2=float(candidate.r2),
        start_idx=int(candidate.i0),
        end_idx=int(candidate.i1),
        start_strain=float(strain[candidate.i0]),
        end_strain=float(strain[candidate.i1]),
        slope_cv=float(candidate.slope_cv),
        zero_strain=float(zero_strain),
        fit_mode=fit_mode,
    )


def _has_detectable_pre_yield_window(candidate: LinearRegionCandidate, *, r2_min: float) -> bool:
    return (
        np.isfinite(candidate.slope)
        and candidate.slope > 0
        and np.isfinite(candidate.r2)
        and candidate.r2 >= r2_min
    )


def _resolve_window_max(window_max, *, yield_cutoff_idx: int | None) -> int:
    usable_points = max(2, 1 if yield_cutoff_idx is None else int(yield_cutoff_idx) + 1)
    if window_max is None:
        return usable_points
    return min(int(window_max), usable_points)


def _is_short_ramp_region(
    strain: np.ndarray,
    *,
    yield_cutoff_idx: int | None,
    window_min: int,
) -> bool:
    if strain.size < 2 or yield_cutoff_idx is None:
        return False

    stop = min(max(int(yield_cutoff_idx), 0), len(strain) - 1)
    usable_points = stop + 1
    if usable_points < COMPRESSION_SHORT_RAMP_WINDOW_MIN:
        return False

    pre_yield_origin = min(0.0, float(np.nanmin(strain[: stop + 1])))
    pre_yield_span = max(0.0, float(strain[stop] - pre_yield_origin))

    return (
        pre_yield_span <= COMPRESSION_SHORT_RAMP_PRE_YIELD_STRAIN_MAX
        and usable_points <= max(int(window_min) * 2, COMPRESSION_SHORT_RAMP_USABLE_POINTS_MAX)
    )


def _search_candidate_stage(
    strain: np.ndarray,
    stress: np.ndarray,
    *,
    window_min: int,
    window_max: int,
    window_step: int,
    r2_min: float,
    slope_cv_max: float,
    zero_strain_max: float,
    start_idx: int,
    end_idx: int,
    fit_mode_override: str | None = None,
):
    search = enumerate_linear_region_candidates(
        strain,
        stress,
        window_min=window_min,
        window_max=window_max,
        window_step=window_step,
        threshold_stress=0.0,
        start_idx=start_idx,
        end_idx=end_idx,
    )

    candidates = []
    has_linear_pre_yield_window = False
    for raw_candidate in search.candidates:
        if _has_detectable_pre_yield_window(raw_candidate, r2_min=r2_min):
            has_linear_pre_yield_window = True

        converted = _convert_candidate(
            raw_candidate,
            search.strain,
            yield_cutoff_idx=end_idx,
            r2_min=r2_min,
            slope_cv_max=slope_cv_max,
            zero_strain_max=zero_strain_max,
            fit_mode_override=fit_mode_override,
        )
        if converted is not None:
            candidates.append(converted)

    candidates.sort(key=_candidate_sort_key)
    return candidates, has_linear_pre_yield_window


def extract_compression_candidates(
    strain,
    stress,
    *,
    window_min=COMPRESSION_CANDIDATE_WINDOW_MIN,
    window_max=COMPRESSION_CANDIDATE_WINDOW_MAX,
    window_step=COMPRESSION_CANDIDATE_WINDOW_STEP,
    r2_min=COMPRESSION_PRIMARY_R2_MIN,
    slope_cv_max=COMPRESSION_SLOPE_CV_MAX,
    zero_strain_max=COMPRESSION_ZERO_STRAIN_MAX,
):
    strain = np.asarray(strain, dtype=float)
    stress = np.asarray(stress, dtype=float)

    if strain.size < 2:
        return [], None, False, False, False

    yield_cutoff_idx, geometric_instability = detect_pre_yield_cutoff(strain, stress)
    capped_window_max = _resolve_window_max(window_max, yield_cutoff_idx=yield_cutoff_idx)
    windup_end_idx = detect_startup_windup_end(
        strain,
        stress,
        max_idx=yield_cutoff_idx,
    )

    attempts = [(windup_end_idx, False)]
    if windup_end_idx > 0:
        attempts.append((0, True))

    has_linear_pre_yield_window = False
    for attempt_start_idx, used_retry in attempts:
        candidates, stage_has_linear = _search_candidate_stage(
            strain,
            stress,
            window_min=window_min,
            window_max=capped_window_max,
            window_step=window_step,
            r2_min=r2_min,
            slope_cv_max=slope_cv_max,
            zero_strain_max=zero_strain_max,
            start_idx=attempt_start_idx,
            end_idx=yield_cutoff_idx,
        )
        has_linear_pre_yield_window = has_linear_pre_yield_window or stage_has_linear
        if candidates:
            return candidates, yield_cutoff_idx, geometric_instability, has_linear_pre_yield_window, used_retry

    if _is_short_ramp_region(
        strain,
        yield_cutoff_idx=yield_cutoff_idx,
        window_min=window_min,
    ):
        short_window_max = _resolve_window_max(
            COMPRESSION_SHORT_RAMP_WINDOW_MAX,
            yield_cutoff_idx=yield_cutoff_idx,
        )
        short_window_min = min(COMPRESSION_SHORT_RAMP_WINDOW_MIN, short_window_max)

        for attempt_start_idx, used_retry in attempts:
            candidates, stage_has_linear = _search_candidate_stage(
                strain,
                stress,
                window_min=short_window_min,
                window_max=short_window_max,
                window_step=COMPRESSION_SHORT_RAMP_WINDOW_STEP,
                r2_min=COMPRESSION_SHORT_RAMP_R2_MIN,
                slope_cv_max=COMPRESSION_SHORT_RAMP_SLOPE_CV_MAX,
                zero_strain_max=COMPRESSION_SHORT_RAMP_ZERO_STRAIN_MAX,
                start_idx=attempt_start_idx,
                end_idx=yield_cutoff_idx,
                fit_mode_override=FIT_MODE_SHORT_RAMP_EXCEPTION,
            )
            has_linear_pre_yield_window = has_linear_pre_yield_window or stage_has_linear
            if candidates:
                return candidates, yield_cutoff_idx, geometric_instability, has_linear_pre_yield_window, used_retry

        relaxed_window_max = _resolve_window_max(
            COMPRESSION_SHORT_RAMP_RELAXED_WINDOW_MAX,
            yield_cutoff_idx=yield_cutoff_idx,
        )
        relaxed_window_min = min(COMPRESSION_SHORT_RAMP_RELAXED_WINDOW_MIN, relaxed_window_max)

        for attempt_start_idx, used_retry in attempts:
            candidates, stage_has_linear = _search_candidate_stage(
                strain,
                stress,
                window_min=relaxed_window_min,
                window_max=relaxed_window_max,
                window_step=COMPRESSION_SHORT_RAMP_RELAXED_WINDOW_STEP,
                r2_min=COMPRESSION_SHORT_RAMP_RELAXED_R2_MIN,
                slope_cv_max=COMPRESSION_SHORT_RAMP_RELAXED_SLOPE_CV_MAX,
                zero_strain_max=COMPRESSION_SHORT_RAMP_RELAXED_ZERO_STRAIN_MAX,
                start_idx=attempt_start_idx,
                end_idx=yield_cutoff_idx,
                fit_mode_override=FIT_MODE_SHORT_RAMP_RELAXED,
            )
            has_linear_pre_yield_window = has_linear_pre_yield_window or stage_has_linear
            if candidates:
                return candidates, yield_cutoff_idx, geometric_instability, has_linear_pre_yield_window, used_retry

    return [], yield_cutoff_idx, geometric_instability, has_linear_pre_yield_window, False


def find_linear_region_properties_V3(
    strain,
    stress,
    *,
    window_min=COMPRESSION_CANDIDATE_WINDOW_MIN,
    window_step=COMPRESSION_CANDIDATE_WINDOW_STEP,
    r2_min=COMPRESSION_PRIMARY_R2_MIN,
    threshold_stress=0.0,
    fallback_min_points=20,
):
    _ = fallback_min_points
    fit = select_compression_fit(
        strain,
        stress,
        window_min=window_min,
        primary_window_step=window_step,
        r2_min=r2_min,
        threshold_stress=threshold_stress,
    )
    return (
        fit["slope"],
        fit["start_idx"],
        fit["end_idx"],
        fit["r2"],
        fit["strain"],
        fit["stress"],
        fit["intercept"],
    )


def _is_confident_compression_fit(slope, r2, *, r2_min=COMPRESSION_PRIMARY_R2_MIN):
    return (
        np.isfinite(slope)
        and slope > 0
        and r2 is not None
        and np.isfinite(r2)
        and r2 >= r2_min
    )


def select_compression_fit(
    strain,
    stress,
    *,
    window_min=COMPRESSION_CANDIDATE_WINDOW_MIN,
    primary_window_step=COMPRESSION_CANDIDATE_WINDOW_STEP,
    r2_min=COMPRESSION_PRIMARY_R2_MIN,
    threshold_stress=0.0,
    fallback_min_points=20,
    retry_window_min=None,
    retry_window_step=None,
    window_max=COMPRESSION_CANDIDATE_WINDOW_MAX,
    zero_strain_max=COMPRESSION_ZERO_STRAIN_MAX,
    slope_cv_max=COMPRESSION_SLOPE_CV_MAX,
):
    _ = fallback_min_points
    _ = retry_window_min
    _ = retry_window_step

    s_clean, t_clean = clean_curve_data(strain, stress)
    s_clean = np.asarray(s_clean, dtype=float)
    t_clean = np.asarray(t_clean, dtype=float)

    if threshold_stress > 0.0 and s_clean.size > 0:
        mask = t_clean >= threshold_stress
        s_clean = s_clean[mask]
        t_clean = t_clean[mask]

    if s_clean.size < 2:
        return {
            "slope": np.nan,
            "start_idx": None,
            "end_idx": None,
            "r2": np.nan,
            "strain": s_clean,
            "stress": t_clean,
            "intercept": np.nan,
            "is_valid": False,
            "fit_mode": "",
            "decision_reason": DECISION_REJECTED_NO_FIT,
            "candidates": [],
            "yield_cutoff_idx": None,
            "geometric_instability": False,
            "used_retry": False,
        }

    candidates, yield_cutoff_idx, geometric_instability, has_linear_pre_yield_window, used_retry = extract_compression_candidates(
        s_clean,
        t_clean,
        window_min=window_min,
        window_max=window_max,
        window_step=primary_window_step,
        r2_min=r2_min,
        zero_strain_max=zero_strain_max,
        slope_cv_max=slope_cv_max,
    )

    if candidates:
        best = candidates[0]
        return {
            "slope": best.slope,
            "start_idx": best.start_idx,
            "end_idx": best.end_idx,
            "r2": best.r2,
            "strain": s_clean,
            "stress": t_clean,
            "intercept": best.intercept,
            "is_valid": True,
            "fit_mode": best.fit_mode,
            "decision_reason": DECISION_ACCEPTED_CLUSTERED,
            "candidates": candidates,
            "yield_cutoff_idx": yield_cutoff_idx,
            "geometric_instability": geometric_instability,
            "used_retry": used_retry,
        }

    return {
        "slope": np.nan,
        "start_idx": None,
        "end_idx": None,
        "r2": np.nan,
        "strain": s_clean,
        "stress": t_clean,
        "intercept": np.nan,
        "is_valid": False,
        "fit_mode": "",
        "decision_reason": (
            DECISION_REJECTED_GEOMETRIC
            if geometric_instability and has_linear_pre_yield_window
            else DECISION_REJECTED_NO_FIT
        ),
        "candidates": [],
        "yield_cutoff_idx": yield_cutoff_idx,
        "geometric_instability": geometric_instability,
        "used_retry": used_retry,
    }
