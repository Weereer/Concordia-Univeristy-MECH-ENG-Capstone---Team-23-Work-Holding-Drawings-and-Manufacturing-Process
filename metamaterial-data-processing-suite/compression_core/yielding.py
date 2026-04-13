from __future__ import annotations

import numpy as np

from shared_core.curve_components import clean_curve_data, find_offset_intersection

from .constants import (
    YIELD_METHOD_FIRST_MAX,
    YIELD_METHOD_PROOF,
    YIELD_METHOD_SOFTENING,
)


def _median_smooth(values: np.ndarray, radius: int = 1) -> np.ndarray:
    if values.size == 0:
        return values

    smoothed = np.empty_like(values, dtype=float)
    for i in range(len(values)):
        lo = max(0, i - radius)
        hi = min(len(values), i + radius + 1)
        smoothed[i] = float(np.median(values[lo:hi]))
    return smoothed


def _proof_stress_intersection(
    strain_clean,
    stress_clean,
    E,
    offset,
    intercept,
):
    if not np.isfinite(E):
        return np.array([], dtype=float), np.array([], dtype=float), np.nan, np.nan

    proof = find_offset_intersection(
        strain_clean,
        stress_clean,
        E,
        offset,
        intercept,
    )
    return (
        proof.offset_strain,
        proof.offset_stress,
        proof.intersection_strain,
        proof.intersection_stress,
    )


def _first_local_maximum_idx(stress, *, start_idx=0):
    stress = np.asarray(stress, dtype=float)
    if stress.size < 3:
        return None

    start = max(int(start_idx), 1)
    smoothed = _median_smooth(stress, radius=1)
    for idx in range(start, len(smoothed) - 1):
        left = smoothed[idx - 1]
        center = smoothed[idx]
        right = smoothed[idx + 1]
        if not np.isfinite(center):
            continue
        if center >= left and center > right:
            return idx
    return None


def determine_compression_yield(
    strain_clean,
    stress_clean,
    E,
    offset,
    linear_start,
    linear_end,
    intercept,
    *,
    yield_cutoff_idx=None,
):
    _ = linear_start
    strain_clean, stress_clean = clean_curve_data(strain_clean, stress_clean)
    proof_offset_strain, proof_offset_stress, proof_strain, proof_stress = _proof_stress_intersection(
        strain_clean,
        stress_clean,
        E,
        offset,
        intercept,
    )

    yield_strain = np.nan
    yield_stress = np.nan
    yield_method = ""
    offset_strain = proof_offset_strain
    offset_stress = proof_offset_stress

    proof_is_usable = (
        np.isfinite(proof_strain)
        and np.isfinite(proof_stress)
        and (
            linear_end is None
            or proof_strain >= float(strain_clean[min(int(linear_end), len(strain_clean) - 1)])
        )
    )
    has_softening_cutoff = (
        yield_cutoff_idx is not None
        and strain_clean.size > 0
        and 0 <= int(yield_cutoff_idx) < len(strain_clean) - 1
    )
    if proof_is_usable:
        yield_strain = float(proof_strain)
        yield_stress = float(proof_stress)
        yield_method = YIELD_METHOD_PROOF
    elif has_softening_cutoff:
        yield_idx = int(yield_cutoff_idx)
        if linear_end is not None:
            yield_idx = max(yield_idx, min(int(linear_end) + 1, len(strain_clean) - 1))
        yield_strain = float(strain_clean[yield_idx])
        yield_stress = float(stress_clean[yield_idx])
        yield_method = YIELD_METHOD_SOFTENING
    else:
        max_idx = _first_local_maximum_idx(
            stress_clean,
            start_idx=0 if linear_end is None else int(linear_end) + 1,
        )
        if max_idx is not None:
            yield_strain = float(strain_clean[max_idx])
            yield_stress = float(stress_clean[max_idx])
            yield_method = YIELD_METHOD_FIRST_MAX
        elif np.isfinite(proof_strain) and np.isfinite(proof_stress):
            yield_strain = float(proof_strain)
            yield_stress = float(proof_stress)
            yield_method = YIELD_METHOD_PROOF

    ultimate_strength = float(np.nanmax(stress_clean)) if np.size(stress_clean) > 0 else np.nan
    return offset_strain, offset_stress, yield_strain, yield_stress, ultimate_strength, yield_method


def offset_intersection(strain_clean, stress_clean, E, offset, linear_start, linear_end, intercept):
    _ = linear_start
    _ = linear_end
    offset_strain, offset_stress, proof_strain, proof_stress = _proof_stress_intersection(
        strain_clean,
        stress_clean,
        E,
        offset,
        intercept,
    )
    ultimate_strength = float(np.nanmax(stress_clean)) if np.size(stress_clean) > 0 else np.nan
    return offset_strain, offset_stress, proof_strain, proof_stress, ultimate_strength
