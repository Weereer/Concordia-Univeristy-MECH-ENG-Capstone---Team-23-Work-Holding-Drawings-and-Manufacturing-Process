from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from shared_core.adaptive_fit import ElasticFitCandidate, ElasticSampleAnalysis
from shear_core.decisions import build_shear_subgroup_band, resolve_shear_subgroup_fit_decisions


def make_candidate(slope_gpa: float, *, r2: float = 0.99, fit_mode: str = "direct") -> ElasticFitCandidate:
    slope = slope_gpa * 1e9
    return ElasticFitCandidate(
        slope=slope,
        intercept=0.0,
        r2=r2,
        start_idx=0,
        end_idx=10,
        start_strain=0.001,
        end_strain=0.010,
        slope_cv=0.1,
        zero_strain=0.0,
        fit_mode=fit_mode,
    )


def make_sample(sample_name: str, *candidates: ElasticFitCandidate) -> ElasticSampleAnalysis:
    return ElasticSampleAnalysis(
        sample_name=sample_name,
        file_path=Path(sample_name),
        material="PETG",
        strain=np.array([0.0, 0.01], dtype=float),
        stress=np.array([0.0, 1.0], dtype=float),
        candidates=list(candidates),
    )


class ShearDecisionTests(unittest.TestCase):
    def test_build_shear_subgroup_band_widens_relative_fallback_band(self):
        band = build_shear_subgroup_band([0.20, 0.24])

        self.assertIsNotNone(band)
        self.assertEqual(band["mode"], "relative")
        self.assertEqual(band["relative_band"], 0.30)

    def test_resolve_shear_subgroup_fit_decisions_accepts_small_group_spread(self):
        sample_a = make_sample("P1SH40 - 1", make_candidate(0.1493))
        sample_b = make_sample("P1SH40 - 4", make_candidate(0.2049))

        resolve_shear_subgroup_fit_decisions([sample_a, sample_b])

        self.assertTrue(sample_a.is_valid)
        self.assertEqual(sample_a.decision_reason, "accepted_clustered")
        self.assertTrue(sample_b.is_valid)
        self.assertEqual(sample_b.decision_reason, "accepted_clustered")


if __name__ == "__main__":
    unittest.main()
