from __future__ import annotations

import numpy as np

from shared_core.adaptive_fit import (
    DECISION_ACCEPTED_CLUSTERED,
    DECISION_ACCEPTED_CORRECTED,
    DECISION_REJECTED_NO_FIT,
    DECISION_REJECTED_OUTLIER,
    FIT_MODE_ALTERNATE,
    FIT_MODE_DIRECT,
    FIT_MODE_STRESS_GATE_TRIMMED,
    ElasticSampleAnalysis,
    build_subgroup_band,
    extract_elastic_candidates,
    is_within_subgroup_band,
)

from .constants import (
    PETG_TENSION_BRITTLE_START_TOLERANCE,
    PETG_TENSION_FALLBACK_MIN_POINTS,
    PETG_TENSION_MEANINGFUL_PROOF_GAP,
    PETG_TENSION_PHYSICAL_OVERRIDE_SLOPE_RATIO_MAX,
    PETG_TENSION_PHYSICAL_OVERRIDE_SLOPE_RATIO_MIN,
    PETG_TENSION_PROOF_ZERO_STRAIN_MAX,
    DECISION_ACCEPTED_PETG_PHYSICAL_OVERRIDE,
    DECISION_ACCEPTED_PETG_RECOVERY,
    FIT_MODE_PETG_SHORT_WINDOW_RECOVERY,
    PETG_TENSION_RECOVERY_MIN_START_DELTA,
    PETG_TENSION_RECOVERY_WINDOW_MIN,
    PETG_TENSION_RECOVERY_WINDOW_STEP,
    TENSION_PRIMARY_R2_MIN,
    TENSION_SLOPE_CV_MAX,
    TENSION_STRESS_GATE_DELTA,
    TENSION_THRESHOLD_STRESS,
    TENSION_YIELD_METHOD_FIRST_MAX,
    TENSION_YIELD_METHOD_PROOF,
    TENSION_ZERO_STRAIN_MAX,
)
from .yielding import determine_tension_yield, find_tension_analysis_end_idx


_PETG_PHYSICAL_OVERRIDE_PEAK_END_TOL = 1.0e-6


def _apply_default_subgroup_resolution(samples: list[ElasticSampleAnalysis]):
    candidate_slopes = [
        sample.candidates[0].slope
        for sample in samples
        if sample.candidates
    ]
    band = build_subgroup_band(candidate_slopes)

    for sample in samples:
        if not sample.candidates:
            sample.selected_candidate = None
            sample.is_valid = False
            sample.fit_mode = ""
            sample.decision_reason = DECISION_REJECTED_NO_FIT
            continue

        top_candidate = sample.candidates[0]
        if band is None or is_within_subgroup_band(top_candidate.slope, band):
            sample.selected_candidate = top_candidate
            sample.is_valid = True
            sample.fit_mode = top_candidate.fit_mode
            sample.decision_reason = DECISION_ACCEPTED_CLUSTERED
            continue

        alternate = next(
            (
                candidate
                for candidate in sample.candidates[1:]
                if is_within_subgroup_band(candidate.slope, band)
            ),
            None,
        )

        if alternate is not None:
            sample.selected_candidate = alternate
            sample.is_valid = True
            sample.fit_mode = FIT_MODE_ALTERNATE
            sample.decision_reason = DECISION_ACCEPTED_CORRECTED
        else:
            sample.selected_candidate = None
            sample.is_valid = False
            sample.fit_mode = ""
            sample.decision_reason = DECISION_REJECTED_OUTLIER

    return band


def _selected_petg_start_strain_median(samples: list[ElasticSampleAnalysis]):
    start_strains = [
        float(sample.selected_candidate.start_strain)
        for sample in samples
        if sample.material.upper() == "PETG"
        and sample.is_valid
        and sample.selected_candidate is not None
    ]
    if not start_strains:
        return np.nan
    return float(np.median(np.asarray(start_strains, dtype=float)))


def _petg_candidate_yield_result(
    sample: ElasticSampleAnalysis,
    candidate,
    yield_result_cache: dict[tuple[int, int], tuple],
):
    cache_key = (id(sample), id(candidate))
    cached = yield_result_cache.get(cache_key)
    if cached is not None:
        return cached

    result = determine_tension_yield(
        sample.strain,
        sample.stress,
        candidate.slope,
        0.002,
        candidate.start_idx,
        candidate.end_idx,
        candidate.intercept,
        material="PETG",
    )
    yield_result_cache[cache_key] = result
    return result


def _petg_candidate_yield_method(sample: ElasticSampleAnalysis, candidate, yield_result_cache: dict[tuple[int, int], tuple]) -> str:
    return _petg_candidate_yield_result(sample, candidate, yield_result_cache)[-1]


def _search_petg_recovery_candidates(
    sample: ElasticSampleAnalysis,
    recovery_candidate_cache: dict[int, list],
):
    sample_id = id(sample)
    cached = recovery_candidate_cache.get(sample_id)
    if cached is not None:
        return cached

    cached = list(
        extract_elastic_candidates(
            sample.strain,
            sample.stress,
            window_min=PETG_TENSION_RECOVERY_WINDOW_MIN,
            window_step=PETG_TENSION_RECOVERY_WINDOW_STEP,
            r2_min=TENSION_PRIMARY_R2_MIN,
            threshold_stress=TENSION_THRESHOLD_STRESS,
            stress_gate_delta=TENSION_STRESS_GATE_DELTA,
            zero_strain_max=TENSION_ZERO_STRAIN_MAX,
            slope_cv_max=TENSION_SLOPE_CV_MAX,
            fallback_min_points=PETG_TENSION_FALLBACK_MIN_POINTS,
        ).candidates
    )
    recovery_candidate_cache[sample_id] = cached
    return cached


def _petg_recovery_sort_key(candidate, band):
    median = float(band["median"]) if band is not None else np.nan
    distance_from_band = abs(candidate.slope - median) if np.isfinite(median) else 0.0
    return (
        distance_from_band,
        candidate.slope_cv,
        -candidate.r2,
        -candidate.window_points,
        candidate.start_strain,
        abs(candidate.zero_strain),
    )


def _find_petg_recovery_candidate(
    sample: ElasticSampleAnalysis,
    band,
    *,
    yield_result_cache: dict[tuple[int, int], tuple],
    recovery_candidate_cache: dict[int, list],
):
    if sample.material.upper() != "PETG" or band is None:
        return None

    recovery_candidates = _search_petg_recovery_candidates(sample, recovery_candidate_cache)
    if not recovery_candidates:
        return None

    in_band_candidates = [
        candidate
        for candidate in recovery_candidates
        if is_within_subgroup_band(candidate.slope, band)
        and _petg_candidate_yield_method(sample, candidate, yield_result_cache) == TENSION_YIELD_METHOD_PROOF
    ]
    if not in_band_candidates:
        return None

    if sample.decision_reason == DECISION_REJECTED_OUTLIER and sample.candidates:
        latest_allowed_start = float(sample.candidates[0].start_strain) - float(PETG_TENSION_RECOVERY_MIN_START_DELTA)
        in_band_candidates = [
            candidate
            for candidate in in_band_candidates
            if float(candidate.start_strain) <= latest_allowed_start
        ]
        if not in_band_candidates:
            return None

    return min(
        in_band_candidates,
        key=lambda candidate: _petg_recovery_sort_key(candidate, band),
    )


def _find_petg_physical_override_candidate(
    sample: ElasticSampleAnalysis,
    *,
    band,
    sibling_start_strain_median: float,
    yield_result_cache: dict[tuple[int, int], tuple],
):
    if sample.material.upper() != "PETG" or sample.is_valid or not sample.candidates:
        return None

    if not np.isfinite(sibling_start_strain_median) or band is None:
        return None

    sibling_slope_median = float(band.get("median", np.nan))
    if not np.isfinite(sibling_slope_median) or sibling_slope_median == 0.0:
        return None

    latest_physical_start = float(sibling_start_strain_median) + float(PETG_TENSION_RECOVERY_MIN_START_DELTA)

    for candidate in sample.candidates:
        if candidate.fit_mode != FIT_MODE_DIRECT:
            continue

        if float(candidate.start_strain) > latest_physical_start:
            continue

        slope_ratio = float(candidate.slope) / sibling_slope_median
        if (
            not np.isfinite(slope_ratio)
            or slope_ratio < float(PETG_TENSION_PHYSICAL_OVERRIDE_SLOPE_RATIO_MIN)
            or slope_ratio > float(PETG_TENSION_PHYSICAL_OVERRIDE_SLOPE_RATIO_MAX)
        ):
            continue

        yield_method = _petg_candidate_yield_method(sample, candidate, yield_result_cache)
        if yield_method == TENSION_YIELD_METHOD_PROOF:
            return candidate

        if yield_method != TENSION_YIELD_METHOD_FIRST_MAX:
            continue

        yield_result = _petg_candidate_yield_result(sample, candidate, yield_result_cache)
        yield_strain = float(yield_result[2]) if np.isfinite(yield_result[2]) else np.nan
        if not np.isfinite(yield_strain):
            continue

        if float(candidate.end_strain) <= yield_strain + _PETG_PHYSICAL_OVERRIDE_PEAK_END_TOL:
            return candidate

    return None


def _petg_analysis_peak_and_end_strain(sample: ElasticSampleAnalysis):
    if sample.strain.size == 0 or sample.stress.size == 0:
        return np.nan, np.nan

    peak_idx = int(np.nanargmax(sample.stress))
    analysis_end_idx = find_tension_analysis_end_idx(sample.stress)
    if analysis_end_idx is None:
        analysis_end_idx = len(sample.strain) - 1
    analysis_end_idx = min(int(analysis_end_idx), len(sample.strain) - 1)
    return float(sample.strain[peak_idx]), float(sample.strain[analysis_end_idx])


def _find_petg_peak_override_candidate(
    sample: ElasticSampleAnalysis,
    *,
    sibling_start_strain_median: float,
    yield_result_cache: dict[tuple[int, int], tuple],
):
    if sample.material.upper() != "PETG" or sample.is_valid or not sample.candidates:
        return None

    if not np.isfinite(sibling_start_strain_median):
        return None

    peak_strain, analysis_end_strain = _petg_analysis_peak_and_end_strain(sample)
    if (
        not np.isfinite(peak_strain)
        or not np.isfinite(analysis_end_strain)
        or analysis_end_strain > peak_strain + float(PETG_TENSION_BRITTLE_START_TOLERANCE)
    ):
        return None

    latest_physical_start = float(sibling_start_strain_median) + float(PETG_TENSION_RECOVERY_MIN_START_DELTA)
    for candidate in sample.candidates:
        if float(candidate.start_strain) > latest_physical_start:
            continue
        if _petg_candidate_yield_method(sample, candidate, yield_result_cache) != TENSION_YIELD_METHOD_FIRST_MAX:
            continue
        return candidate

    return None


def _petg_proof_candidate_sort_key(candidate):
    fit_mode_rank = 0
    if candidate.fit_mode == FIT_MODE_STRESS_GATE_TRIMMED:
        fit_mode_rank = 1
    elif candidate.fit_mode != FIT_MODE_DIRECT:
        fit_mode_rank = 2
    return (
        candidate.start_strain,
        fit_mode_rank,
        -candidate.r2,
        candidate.slope_cv,
        -candidate.window_points,
        abs(candidate.zero_strain),
    )


def _dedupe_petg_candidates(candidates):
    unique_candidates = []
    seen_candidates = set()
    for candidate in candidates:
        if candidate is None or candidate in seen_candidates:
            continue
        seen_candidates.add(candidate)
        unique_candidates.append(candidate)
    return unique_candidates


def _find_petg_proof_candidate(
    sample: ElasticSampleAnalysis,
    band,
    *,
    yield_result_cache: dict[tuple[int, int], tuple],
    recovery_candidate_cache: dict[int, list],
):
    if sample.material.upper() != "PETG" or sample.selected_candidate is None:
        return None

    peak_strain, _ = _petg_analysis_peak_and_end_strain(sample)
    if not np.isfinite(peak_strain):
        return None

    selected_candidate = sample.selected_candidate
    candidate_pool = _dedupe_petg_candidates(
        [
            selected_candidate,
            *sample.candidates,
            *_search_petg_recovery_candidates(sample, recovery_candidate_cache),
        ]
    )
    in_band_candidates = [
        candidate
        for candidate in candidate_pool
        if band is None or is_within_subgroup_band(candidate.slope, band)
    ]
    if not in_band_candidates:
        return None

    ordered_candidates = sorted(
        in_band_candidates,
        key=_petg_proof_candidate_sort_key,
    )
    strict_proof_candidates = []
    fallback_proof_candidates = []
    for candidate in ordered_candidates:
        if _petg_candidate_yield_method(sample, candidate, yield_result_cache) != TENSION_YIELD_METHOD_PROOF:
            continue

        yield_result = _petg_candidate_yield_result(sample, candidate, yield_result_cache)
        yield_strain = float(yield_result[2]) if np.isfinite(yield_result[2]) else np.nan
        if not np.isfinite(yield_strain):
            continue
        if peak_strain - yield_strain < float(PETG_TENSION_MEANINGFUL_PROOF_GAP):
            continue

        if abs(float(candidate.zero_strain)) <= float(PETG_TENSION_PROOF_ZERO_STRAIN_MAX):
            strict_proof_candidates.append(candidate)
        else:
            fallback_proof_candidates.append(candidate)

    if strict_proof_candidates:
        return strict_proof_candidates[0]
    if fallback_proof_candidates:
        return fallback_proof_candidates[0]

    return None


def resolve_subgroup_fit_decisions(samples: list[ElasticSampleAnalysis]):
    band = _apply_default_subgroup_resolution(samples)
    if band is None:
        return samples

    sibling_start_strain_median = _selected_petg_start_strain_median(samples)
    yield_result_cache: dict[tuple[int, int], tuple] = {}
    recovery_candidate_cache: dict[int, list] = {}

    for sample in samples:
        if sample.is_valid or sample.material.upper() != "PETG":
            continue

        recovery_candidate = _find_petg_recovery_candidate(
            sample,
            band,
            yield_result_cache=yield_result_cache,
            recovery_candidate_cache=recovery_candidate_cache,
        )
        if recovery_candidate is None:
            physical_override = _find_petg_physical_override_candidate(
                sample,
                band=band,
                sibling_start_strain_median=sibling_start_strain_median,
                yield_result_cache=yield_result_cache,
            )
            if physical_override is not None:
                sample.selected_candidate = physical_override
                sample.is_valid = True
                sample.fit_mode = physical_override.fit_mode
                sample.decision_reason = DECISION_ACCEPTED_PETG_PHYSICAL_OVERRIDE
                continue

            peak_override = _find_petg_peak_override_candidate(
                sample,
                sibling_start_strain_median=sibling_start_strain_median,
                yield_result_cache=yield_result_cache,
            )
            if peak_override is None:
                continue

            sample.selected_candidate = peak_override
            sample.is_valid = True
            sample.fit_mode = peak_override.fit_mode
            sample.decision_reason = DECISION_ACCEPTED_PETG_PHYSICAL_OVERRIDE
            continue

        sample.selected_candidate = recovery_candidate
        sample.is_valid = True
        sample.fit_mode = FIT_MODE_PETG_SHORT_WINDOW_RECOVERY
        sample.decision_reason = DECISION_ACCEPTED_PETG_RECOVERY

    for sample in samples:
        if sample.material.upper() != "PETG" or not sample.is_valid or sample.selected_candidate is None:
            continue

        proof_candidate = _find_petg_proof_candidate(
            sample,
            band,
            yield_result_cache=yield_result_cache,
            recovery_candidate_cache=recovery_candidate_cache,
        )
        if proof_candidate is None or proof_candidate == sample.selected_candidate:
            continue

        sample.selected_candidate = proof_candidate
        sample.fit_mode = proof_candidate.fit_mode

    return samples
