from __future__ import annotations

import numpy as np

from shared_core.adaptive_fit import build_subgroup_band, is_within_subgroup_band

from .constants import (
    COMPRESSION_MANUAL_ACCEPT_TOP_CANDIDATES,
    COMPRESSION_MANUAL_FIT_WINDOWS,
    DECISION_ACCEPTED_CLUSTERED,
    DECISION_ACCEPTED_CORRECTED,
    DECISION_ACCEPTED_EXCEPTION,
    DECISION_REJECTED_GEOMETRIC,
    DECISION_REJECTED_NO_FIT,
    DECISION_REJECTED_OUTLIER,
    FIT_MODE_ALTERNATE,
    FIT_MODE_MANUAL_EXCEPTION,
)
from .models import CompressionFitCandidate, CompressionSampleAnalysis
from .selection import _safe_linear_fit


def _build_manual_exception_candidate(sample: CompressionSampleAnalysis):
    window = COMPRESSION_MANUAL_FIT_WINDOWS.get(sample.sample_name)
    if window is None:
        return None

    strain = np.asarray(sample.strain, dtype=float)
    stress = np.asarray(sample.stress, dtype=float)
    if strain.size < 2 or stress.size != strain.size:
        return None

    start_target, end_target = window
    start_idx = int(np.argmin(np.abs(strain - float(start_target))))
    end_idx = int(np.argmin(np.abs(strain - float(end_target))))
    if end_idx <= start_idx:
        return None

    fit = _safe_linear_fit(strain[start_idx : end_idx + 1], stress[start_idx : end_idx + 1])
    if fit is None:
        return None

    slope, intercept, r2 = fit
    if not np.isfinite(slope) or slope <= 0:
        return None

    zero_strain = float(-intercept / slope) if slope != 0 else float("inf")
    return CompressionFitCandidate(
        slope=float(slope),
        intercept=float(intercept),
        r2=float(r2),
        start_idx=start_idx,
        end_idx=end_idx,
        start_strain=float(strain[start_idx]),
        end_strain=float(strain[end_idx]),
        slope_cv=0.0,
        zero_strain=zero_strain,
        fit_mode=FIT_MODE_MANUAL_EXCEPTION,
    )


def _apply_manual_top_candidate_exception(sample: CompressionSampleAnalysis):
    if sample.is_valid or sample.sample_name not in COMPRESSION_MANUAL_ACCEPT_TOP_CANDIDATES:
        return False
    if not sample.candidates:
        return False

    top_candidate = sample.candidates[0]
    sample.selected_candidate = top_candidate
    sample.is_valid = True
    sample.fit_mode = top_candidate.fit_mode
    sample.decision_reason = DECISION_ACCEPTED_EXCEPTION
    return True


def resolve_subgroup_fit_decisions(samples: list[CompressionSampleAnalysis]):
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
            sample.decision_reason = (
                DECISION_REJECTED_GEOMETRIC
                if sample.geometric_instability
                else DECISION_REJECTED_NO_FIT
            )
        else:
            top_candidate = sample.candidates[0]
            if band is None or is_within_subgroup_band(top_candidate.slope, band):
                sample.selected_candidate = top_candidate
                sample.is_valid = True
                sample.fit_mode = top_candidate.fit_mode
                sample.decision_reason = DECISION_ACCEPTED_CLUSTERED
            else:
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

        if _apply_manual_top_candidate_exception(sample):
            continue

        manual_candidate = _build_manual_exception_candidate(sample)
        if manual_candidate is not None and not sample.is_valid:
            sample.selected_candidate = manual_candidate
            sample.is_valid = True
            sample.fit_mode = manual_candidate.fit_mode
            sample.decision_reason = DECISION_ACCEPTED_EXCEPTION

    return samples
