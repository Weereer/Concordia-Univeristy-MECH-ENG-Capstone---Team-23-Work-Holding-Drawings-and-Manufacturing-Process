from __future__ import annotations

import numpy as np

from shared_core.curve_components import clean_curve_data, find_offset_intersection

from .constants import (
    PETG_TENSION_BRITTLE_START_TOLERANCE,
    TENSION_YIELD_METHOD_FIRST_MAX,
    TENSION_YIELD_METHOD_LOCAL_PEAK,
    TENSION_YIELD_METHOD_PROOF,
)


_FAILURE_STEP_DROP_FRACTION = 0.25
_FAILURE_NEAR_ZERO_TAIL_FRACTION = 0.10
_FAILURE_RECOVERY_RATIO_MIN = 0.75
_PROOF_PEAK_STRAIN_TOL = 1.0e-6


def _proof_search_start_strain(strain_clean, linear_start, linear_end):
    if linear_end is not None and 0 <= int(linear_end) < len(strain_clean):
        return float(strain_clean[int(linear_end)])
    if linear_start is not None and 0 <= int(linear_start) < len(strain_clean):
        return float(strain_clean[int(linear_start)])
    return None


def has_tension_recovery_after_index(
    stress_clean: np.ndarray,
    start_idx: int,
    *,
    reference_stress: float,
):
    stress_clean = np.asarray(stress_clean, dtype=float)
    if stress_clean.size == 0:
        return False

    if not np.isfinite(reference_stress) or reference_stress <= 0.0:
        return False

    start = max(int(start_idx), 0)
    if start >= stress_clean.size:
        return False

    tail = stress_clean[start:]
    finite_tail = tail[np.isfinite(tail)]
    if finite_tail.size == 0:
        return False

    recovered_peak = float(np.nanmax(finite_tail))
    return recovered_peak >= _FAILURE_RECOVERY_RATIO_MIN * float(reference_stress)


def find_tension_analysis_end_idx(stress_clean: np.ndarray):
    stress_clean = np.asarray(stress_clean, dtype=float)
    if stress_clean.size == 0:
        return None
    if stress_clean.size == 1:
        return 0

    ultimate_idx = int(np.nanargmax(stress_clean))
    ultimate_strength = float(stress_clean[ultimate_idx])
    if not np.isfinite(ultimate_strength) or ultimate_strength <= 0.0:
        return len(stress_clean) - 1

    catastrophic_step_drop = -_FAILURE_STEP_DROP_FRACTION * ultimate_strength
    near_zero_tail_limit = _FAILURE_NEAR_ZERO_TAIL_FRACTION * ultimate_strength
    tail_max = np.maximum.accumulate(stress_clean[::-1])[::-1]

    for idx in range(ultimate_idx + 1, len(stress_clean)):
        if float(stress_clean[idx] - stress_clean[idx - 1]) <= catastrophic_step_drop:
            if has_tension_recovery_after_index(
                stress_clean,
                idx,
                reference_stress=float(stress_clean[idx - 1]),
            ):
                continue
            return max(idx - 1, ultimate_idx)
        if float(tail_max[idx]) <= near_zero_tail_limit:
            return max(idx - 1, ultimate_idx)

    return len(stress_clean) - 1


def _proof_yield(strain_clean, stress_clean, E, offset, linear_start, linear_end, intercept):
    if not np.isfinite(E):
        return np.array([], dtype=float), np.array([], dtype=float), np.nan, np.nan

    proof = find_offset_intersection(
        strain_clean,
        stress_clean,
        E,
        offset,
        intercept,
        min_strain=_proof_search_start_strain(strain_clean, linear_start, linear_end),
    )
    return (
        proof.offset_strain,
        proof.offset_stress,
        proof.intersection_strain,
        proof.intersection_stress,
    )


def _peak_yield(strain_clean, stress_clean, *, linear_end):
    if strain_clean.size == 0 or stress_clean.size == 0:
        return np.nan, np.nan

    if linear_end is not None:
        start_idx = min(int(linear_end) + 1, len(stress_clean) - 1)
    else:
        start_idx = 0

    if start_idx >= len(stress_clean):
        return np.nan, np.nan

    peak_offset = int(np.nanargmax(stress_clean[start_idx:]))
    peak_idx = start_idx + peak_offset
    return float(strain_clean[peak_idx]), float(stress_clean[peak_idx])


def _first_max_yield(strain_clean, stress_clean):
    if strain_clean.size == 0 or stress_clean.size == 0:
        return np.nan, np.nan

    peak_idx = int(np.nanargmax(stress_clean))
    return float(strain_clean[peak_idx]), float(stress_clean[peak_idx])


def determine_tension_yield(
    strain_clean,
    stress_clean,
    E,
    offset,
    linear_start,
    linear_end,
    intercept,
    *,
    material: str = "",
):
    strain_clean, stress_clean = clean_curve_data(strain_clean, stress_clean)
    analysis_end_idx = find_tension_analysis_end_idx(stress_clean)
    if analysis_end_idx is None:
        analysis_strain = strain_clean
        analysis_stress = stress_clean
    else:
        analysis_strain = strain_clean[: analysis_end_idx + 1]
        analysis_stress = stress_clean[: analysis_end_idx + 1]

    offset_strain, offset_stress, proof_strain, proof_stress = _proof_yield(
        analysis_strain,
        analysis_stress,
        E,
        offset,
        linear_start,
        linear_end,
        intercept,
    )

    yield_strain = np.nan
    yield_stress = np.nan
    yield_method = ""
    material_upper = material.upper()
    first_max_strain = np.nan
    first_max_stress = np.nan
    if material_upper == "PETG" and analysis_strain.size > 0:
        first_max_strain, first_max_stress = _first_max_yield(
            analysis_strain,
            analysis_stress,
        )
    petg_peak_locked = (
        material_upper == "PETG"
        and np.isfinite(first_max_strain)
        and analysis_strain.size > 0
        and float(analysis_strain[-1]) <= float(first_max_strain) + float(PETG_TENSION_BRITTLE_START_TOLERANCE)
    )
    proof_lower_bound = None
    if linear_end is not None and analysis_strain.size > 0:
        proof_lower_bound = float(analysis_strain[min(int(linear_end), len(analysis_strain) - 1)])

    proof_is_usable = (
        np.isfinite(proof_strain)
        and np.isfinite(proof_stress)
        and (proof_lower_bound is None or proof_strain > proof_lower_bound)
        and not petg_peak_locked
        and (
            material_upper != "PETG"
            or not np.isfinite(first_max_strain)
            or proof_strain <= first_max_strain + _PROOF_PEAK_STRAIN_TOL
        )
    )

    if proof_is_usable:
        yield_strain = float(proof_strain)
        yield_stress = float(proof_stress)
        yield_method = TENSION_YIELD_METHOD_PROOF
    elif material_upper == "PETG":
        if np.isfinite(first_max_strain) and np.isfinite(first_max_stress):
            yield_strain = float(first_max_strain)
            yield_stress = float(first_max_stress)
            yield_method = TENSION_YIELD_METHOD_FIRST_MAX
    elif material_upper == "NYLON" and analysis_strain.size > 0:
        peak_strain, peak_stress = _peak_yield(
            analysis_strain,
            analysis_stress,
            linear_end=linear_end,
        )
        if np.isfinite(peak_strain) and np.isfinite(peak_stress):
            yield_strain = peak_strain
            yield_stress = peak_stress
            yield_method = TENSION_YIELD_METHOD_LOCAL_PEAK

    ultimate_strength = float(np.nanmax(analysis_stress)) if analysis_stress.size > 0 else np.nan
    finite_idx = np.where(np.isfinite(analysis_stress))[0]
    fracture_strength = float(analysis_stress[finite_idx[-1]]) if finite_idx.size > 0 else np.nan

    return (
        offset_strain,
        offset_stress,
        yield_strain,
        yield_stress,
        ultimate_strength,
        fracture_strength,
        yield_method,
    )
