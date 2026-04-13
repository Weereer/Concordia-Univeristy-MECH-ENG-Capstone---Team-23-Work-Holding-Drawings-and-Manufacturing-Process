from __future__ import annotations

from shared_core.adaptive_fit import (
    DECISION_ACCEPTED_CLUSTERED,
    DECISION_ACCEPTED_CORRECTED,
    DECISION_REJECTED_NO_FIT,
    DECISION_REJECTED_OUTLIER,
    FIT_MODE_ALTERNATE,
    ElasticSampleAnalysis,
    build_subgroup_band,
    is_within_subgroup_band,
)

from .constants import SHEAR_SUBGROUP_RELATIVE_BAND


def build_shear_subgroup_band(candidate_slopes):
    band = build_subgroup_band(candidate_slopes)
    if band is None or band.get("mode") != "relative":
        return band

    adjusted_band = dict(band)
    adjusted_band["relative_band"] = float(SHEAR_SUBGROUP_RELATIVE_BAND)
    return adjusted_band


def resolve_shear_subgroup_fit_decisions(samples: list[ElasticSampleAnalysis]):
    candidate_slopes = [
        sample.candidates[0].slope
        for sample in samples
        if sample.candidates
    ]
    band = build_shear_subgroup_band(candidate_slopes)

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
