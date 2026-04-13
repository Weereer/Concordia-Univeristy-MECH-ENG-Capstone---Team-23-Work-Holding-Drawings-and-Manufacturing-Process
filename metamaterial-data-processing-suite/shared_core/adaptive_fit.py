from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .curve_components import clean_curve_data
from .linear_region import enumerate_linear_region_candidates


DIAGNOSTIC_HEADERS = [
    "Fit Start Strain",
    "Fit End Strain",
    "Fit Mode",
    "Decision Reason",
]

FIT_MODE_DIRECT = "direct"
FIT_MODE_STRESS_GATE_TRIMMED = "stress_gate_trimmed"
FIT_MODE_SHORT_WINDOW_FALLBACK = "short_window_fallback"
FIT_MODE_ALTERNATE = "alternate_candidate"

DECISION_ACCEPTED_CLUSTERED = "accepted_clustered"
DECISION_ACCEPTED_CORRECTED = "accepted_corrected"
DECISION_REJECTED_NO_FIT = "rejected_no_fit"
DECISION_REJECTED_OUTLIER = "rejected_sibling_outlier"

SUBGROUP_MODIFIED_ZSCORE_MAX = 3.5
SUBGROUP_FALLBACK_RELATIVE_BAND = 0.15
REJECTED_NAME_MARKER = " - Rejected - "
NYLON_CANDIDATE_END_STRAIN_BANDS = (0.60, 0.75, 0.90)


@dataclass(frozen=True)
class ElasticFitCandidate:
    slope: float
    intercept: float
    r2: float
    start_idx: int
    end_idx: int
    start_strain: float
    end_strain: float
    slope_cv: float
    zero_strain: float
    fit_mode: str

    @property
    def window_points(self) -> int:
        return self.end_idx - self.start_idx + 1


@dataclass(frozen=True)
class ElasticFitSearchResult:
    strain: np.ndarray
    stress: np.ndarray
    candidates: tuple[ElasticFitCandidate, ...]
    gate_start_idx: int | None = None


@dataclass
class ElasticSampleAnalysis:
    sample_name: str
    file_path: Path
    material: str
    strain: np.ndarray
    stress: np.ndarray
    candidates: list[ElasticFitCandidate] = field(default_factory=list)
    selected_candidate: ElasticFitCandidate | None = None
    is_valid: bool = False
    fit_mode: str = ""
    decision_reason: str = ""


def canonical_sample_name(sample_name) -> str:
    if sample_name is None:
        return ""
    text = str(sample_name).strip()
    marker_index = text.find(REJECTED_NAME_MARKER)
    if marker_index >= 0:
        return text[:marker_index].strip()
    return text


def format_sample_display_name(sample_name: str, is_valid: bool, decision_reason: str) -> str:
    canonical_name = canonical_sample_name(sample_name)
    if is_valid or not str(decision_reason).startswith("rejected_"):
        return canonical_name
    return f"{canonical_name}{REJECTED_NAME_MARKER}{decision_reason}"


def result_graph_variants(graph_dir: Path, sample_name: str) -> list[Path]:
    canonical_name = canonical_sample_name(sample_name)
    candidate_paths = {graph_dir / f"{canonical_name}.html"}
    candidate_paths.update(graph_dir.glob(f"{canonical_name}{REJECTED_NAME_MARKER}*.html"))
    return sorted(candidate_paths)


def cleanup_stale_result_graphs(graph_path: Path, sample_name: str):
    for candidate_path in result_graph_variants(graph_path.parent, sample_name):
        if candidate_path == graph_path or not candidate_path.exists():
            continue
        candidate_path.unlink()


def build_subgroup_band(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return None

    median = float(np.median(values))
    if values.size >= 4:
        mad = float(np.median(np.abs(values - median)))
        if np.isfinite(mad) and mad > 0:
            return {
                "median": median,
                "mad": mad,
                "mode": "mad",
            }

    return {
        "median": median,
        "mode": "relative",
        "relative_band": SUBGROUP_FALLBACK_RELATIVE_BAND,
    }


def is_within_subgroup_band(value: float, band) -> bool:
    if band is None or not np.isfinite(value):
        return False

    if band["mode"] == "mad":
        score = 0.6745 * (value - band["median"]) / band["mad"]
        return abs(score) <= SUBGROUP_MODIFIED_ZSCORE_MAX

    median = band["median"]
    if not np.isfinite(median):
        return False
    if median == 0:
        return abs(value) <= band["relative_band"]

    deviation = abs(value - median) / abs(median)
    return deviation <= band["relative_band"]


def baseline_stress_rise_start_idx(stress, *, stress_rise_threshold: float):
    stress = np.asarray(stress, dtype=float)
    if stress.size == 0:
        return None
    if not np.isfinite(stress_rise_threshold) or stress_rise_threshold <= 0:
        return 0

    finite_idx = np.flatnonzero(np.isfinite(stress))
    if finite_idx.size == 0:
        return None

    baseline_idx = int(finite_idx[0])
    baseline_stress = float(stress[baseline_idx])
    target_stress = baseline_stress + float(stress_rise_threshold)
    rise_idx = np.flatnonzero(stress >= target_stress)
    if rise_idx.size == 0:
        return None

    return int(rise_idx[0])


def _candidate_sort_key(candidate: ElasticFitCandidate):
    return (
        abs(candidate.zero_strain),
        candidate.slope_cv,
        -candidate.window_points,
        -candidate.r2,
        candidate.start_strain,
    )


def rerank_candidates_by_end_strain_bands(
    candidates,
    *,
    peak_strain: float,
    band_limits=NYLON_CANDIDATE_END_STRAIN_BANDS,
):
    candidate_list = list(candidates)
    if not candidate_list:
        return []
    if not np.isfinite(peak_strain) or peak_strain <= 0.0:
        return sorted(candidate_list, key=_candidate_sort_key)

    grouped_candidates = [[] for _ in range(len(tuple(band_limits)) + 1)]
    for candidate in candidate_list:
        end_ratio = float(candidate.end_strain) / float(peak_strain)
        band_index = len(grouped_candidates) - 1
        for idx, limit in enumerate(band_limits):
            if end_ratio <= float(limit):
                band_index = idx
                break
        grouped_candidates[band_index].append(candidate)

    reranked = []
    for band_candidates in grouped_candidates:
        if not band_candidates:
            continue
        reranked.extend(sorted(band_candidates, key=_candidate_sort_key))

    return reranked


def _map_index(full_strain: np.ndarray, target_strain: float) -> int:
    if full_strain.size == 0:
        return 0
    return int(np.argmin(np.abs(full_strain - float(target_strain))))


def _convert_candidate(
    raw_candidate,
    fit_strain: np.ndarray,
    full_strain: np.ndarray,
    *,
    r2_min: float,
    zero_strain_max: float,
    slope_cv_max: float,
    fit_mode: str,
):
    slope = float(raw_candidate.slope)
    intercept = float(raw_candidate.intercept)
    r2 = float(raw_candidate.r2)
    slope_cv = float(raw_candidate.slope_cv)
    zero_strain = float(raw_candidate.zero_stress_strain)

    if not np.isfinite(slope) or slope <= 0:
        return None
    if not np.isfinite(r2) or r2 < float(r2_min):
        return None
    if not np.isfinite(slope_cv) or slope_cv > float(slope_cv_max):
        return None
    if not np.isfinite(zero_strain) or abs(zero_strain) > float(zero_strain_max):
        return None

    start_strain = float(fit_strain[raw_candidate.i0])
    end_strain = float(fit_strain[raw_candidate.i1])
    start_idx = _map_index(full_strain, start_strain)
    end_idx = _map_index(full_strain, end_strain)
    if end_idx < start_idx:
        return None

    return ElasticFitCandidate(
        slope=slope,
        intercept=intercept,
        r2=r2,
        start_idx=start_idx,
        end_idx=end_idx,
        start_strain=float(full_strain[start_idx]),
        end_strain=float(full_strain[end_idx]),
        slope_cv=slope_cv,
        zero_strain=zero_strain,
        fit_mode=fit_mode,
    )


def _enumerate_candidates(
    full_strain: np.ndarray,
    fit_strain: np.ndarray,
    fit_stress: np.ndarray,
    *,
    window_min: int,
    window_max: int | None,
    window_step: int,
    start_idx: int,
    r2_min: float,
    zero_strain_max: float,
    slope_cv_max: float,
    fit_mode: str,
):
    if fit_strain.size < 2 or fit_stress.size != fit_strain.size:
        return []

    search = enumerate_linear_region_candidates(
        fit_strain,
        fit_stress,
        window_min=window_min,
        window_max=window_max,
        window_step=window_step,
        threshold_stress=0.0,
        start_idx=start_idx,
    )

    candidates = []
    for raw_candidate in search.candidates:
        converted = _convert_candidate(
            raw_candidate,
            fit_strain,
            full_strain,
            r2_min=r2_min,
            zero_strain_max=zero_strain_max,
            slope_cv_max=slope_cv_max,
            fit_mode=fit_mode,
        )
        if converted is not None:
            candidates.append(converted)

    candidates.sort(key=_candidate_sort_key)
    return candidates


def _resolve_short_window_search(
    *,
    requested_window_min: int,
    requested_window_step: int,
    available_points: int,
    fallback_min_points: int,
):
    effective_window_min = max(int(requested_window_min), 2)
    effective_window_step = max(int(requested_window_step), 1)
    used_short_window_fallback = False

    fallback_floor = max(int(fallback_min_points), 2)
    if available_points < effective_window_min and available_points >= fallback_floor:
        effective_window_min = fallback_floor
        effective_window_step = min(effective_window_step, 5)
        used_short_window_fallback = True

    return effective_window_min, effective_window_step, used_short_window_fallback


def extract_elastic_candidates(
    strain,
    stress,
    *,
    window_min: int,
    window_step: int,
    r2_min: float,
    threshold_stress: float,
    stress_gate_delta: float,
    zero_strain_max: float,
    slope_cv_max: float,
    window_max: int | None = None,
    fallback_min_points: int = 20,
):
    full_strain, full_stress = clean_curve_data(strain, stress)
    full_strain = np.asarray(full_strain, dtype=float)
    full_stress = np.asarray(full_stress, dtype=float)

    if full_strain.size < 2:
        return ElasticFitSearchResult(
            strain=full_strain,
            stress=full_stress,
            candidates=tuple(),
            gate_start_idx=None,
        )

    fit_strain = full_strain
    fit_stress = full_stress
    if float(threshold_stress) > 0.0:
        mask = fit_stress >= float(threshold_stress)
        fit_strain = fit_strain[mask]
        fit_stress = fit_stress[mask]

    if fit_strain.size < 2:
        return ElasticFitSearchResult(
            strain=full_strain,
            stress=full_stress,
            candidates=tuple(),
            gate_start_idx=None,
        )

    effective_window_min, effective_window_step, used_short_window_fallback = _resolve_short_window_search(
        requested_window_min=window_min,
        requested_window_step=window_step,
        available_points=int(fit_strain.size),
        fallback_min_points=fallback_min_points,
    )

    direct_fit_mode = FIT_MODE_SHORT_WINDOW_FALLBACK if used_short_window_fallback else FIT_MODE_DIRECT
    gated_fit_mode = FIT_MODE_SHORT_WINDOW_FALLBACK if used_short_window_fallback else FIT_MODE_STRESS_GATE_TRIMMED

    direct_candidates = _enumerate_candidates(
        full_strain,
        fit_strain,
        fit_stress,
        window_min=effective_window_min,
        window_max=window_max,
        window_step=effective_window_step,
        start_idx=0,
        r2_min=r2_min,
        zero_strain_max=zero_strain_max,
        slope_cv_max=slope_cv_max,
        fit_mode=direct_fit_mode,
    )

    gate_start_idx = baseline_stress_rise_start_idx(
        fit_stress,
        stress_rise_threshold=float(stress_gate_delta),
    )

    gated_candidates = []
    if gate_start_idx is not None and gate_start_idx > 0:
        gated_candidates = _enumerate_candidates(
            full_strain,
            fit_strain,
            fit_stress,
            window_min=effective_window_min,
            window_max=window_max,
            window_step=effective_window_step,
            start_idx=gate_start_idx,
            r2_min=r2_min,
            zero_strain_max=zero_strain_max,
            slope_cv_max=slope_cv_max,
            fit_mode=gated_fit_mode,
        )

    candidates = tuple(direct_candidates + gated_candidates)
    full_gate_start_idx = None
    if gate_start_idx is not None and gate_start_idx > 0:
        full_gate_start_idx = _map_index(full_strain, float(fit_strain[gate_start_idx]))

    return ElasticFitSearchResult(
        strain=full_strain,
        stress=full_stress,
        candidates=candidates,
        gate_start_idx=full_gate_start_idx,
    )


def resolve_subgroup_fit_decisions(samples: list[ElasticSampleAnalysis]):
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

    return samples
