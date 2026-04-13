from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OffsetIntersection:
    offset_strain: np.ndarray
    offset_stress: np.ndarray
    intersection_strain: float
    intersection_stress: float


def first_stuck_index(
    x: np.ndarray,
    *,
    tol_strain: float,
    window: int,
    min_start: int,
):
    values = np.asarray(x, dtype=float)
    delta = np.abs(np.diff(values))
    stuck = np.isfinite(delta) & (delta < tol_strain)

    run = 0
    for index, is_stuck in enumerate(stuck):
        if index < min_start:
            run = 0
            continue
        if is_stuck:
            run += 1
            if run >= window:
                return index - window + 1
        else:
            run = 0

    return None


def build_offset_curve(strain: np.ndarray, slope: float, offset: float, intercept: float):
    strain_array = np.asarray(strain, dtype=float)
    return strain_array, slope * (strain_array - offset) + intercept


def clean_curve_data(strain, stress):
    strain_array = np.asarray(strain, dtype=float)
    stress_array = np.asarray(stress, dtype=float)

    mask = np.isfinite(strain_array) & np.isfinite(stress_array)
    strain_array = strain_array[mask]
    stress_array = stress_array[mask]

    if strain_array.size == 0:
        return strain_array, stress_array

    order = np.argsort(strain_array)
    strain_array = strain_array[order]
    stress_array = stress_array[order]

    unique_mask = np.diff(strain_array, prepend=strain_array[0] - 1.0) != 0
    return strain_array[unique_mask], stress_array[unique_mask]


def find_curve_intersection(
    primary_strain,
    primary_stress,
    secondary_strain,
    secondary_stress,
    *,
    min_strain: float | None = None,
    resolution: int = 10000,
):
    primary_strain, primary_stress = clean_curve_data(primary_strain, primary_stress)
    secondary_strain, secondary_stress = clean_curve_data(secondary_strain, secondary_stress)

    if primary_strain.size == 0 or secondary_strain.size == 0:
        return float("nan"), float("nan")

    strain_min = max(float(primary_strain.min()), float(secondary_strain.min()))
    strain_max = min(float(primary_strain.max()), float(secondary_strain.max()))
    if min_strain is not None and np.isfinite(min_strain):
        strain_min = max(strain_min, float(min_strain))
    if strain_max <= strain_min:
        return float("nan"), float("nan")

    from scipy.interpolate import interp1d

    strain_common = np.linspace(strain_min, strain_max, max(int(resolution), 2))
    primary_interp = interp1d(
        primary_strain,
        primary_stress,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
    )
    secondary_interp = interp1d(
        secondary_strain,
        secondary_stress,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
    )
    diff = primary_interp(strain_common) - secondary_interp(strain_common)

    zero_hits = np.where(np.isclose(diff, 0.0))[0]
    if zero_hits.size > 0:
        intersection_strain = float(strain_common[int(zero_hits[0])])
        intersection_stress = float(np.interp(intersection_strain, primary_strain, primary_stress))
        return intersection_strain, intersection_stress

    crossings = np.where(np.sign(diff[:-1]) != np.sign(diff[1:]))[0]
    if crossings.size == 0:
        return float("nan"), float("nan")

    left_index = int(crossings[0])
    x0 = float(strain_common[left_index])
    x1 = float(strain_common[left_index + 1])
    y0 = float(diff[left_index])
    y1 = float(diff[left_index + 1])

    if y1 == y0:
        intersection_strain = x0
    else:
        intersection_strain = x0 - y0 * (x1 - x0) / (y1 - y0)

    intersection_stress = float(np.interp(intersection_strain, primary_strain, primary_stress))
    return intersection_strain, intersection_stress


def find_offset_intersection(
    strain,
    stress,
    slope: float,
    offset: float,
    intercept: float,
    *,
    min_strain: float | None = None,
):
    offset_strain, offset_stress = build_offset_curve(strain, slope, offset, intercept)
    clean_strain, clean_stress = clean_curve_data(strain, stress)
    clean_offset_strain, clean_offset_stress = clean_curve_data(offset_strain, offset_stress)
    intersection_strain, intersection_stress = find_curve_intersection(
        clean_strain,
        clean_stress,
        clean_offset_strain,
        clean_offset_stress,
        min_strain=min_strain,
    )
    return OffsetIntersection(
        offset_strain=clean_offset_strain,
        offset_stress=clean_offset_stress,
        intersection_strain=float(intersection_strain),
        intersection_stress=float(intersection_stress),
    )
