from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from petg_tension_core import (
    PETG_TENSION_GRAPH_VARIANTS,
    PETG_TENSION_OFFSET_STRAIN,
    describe_petg_tension_graph_system,
    render_petg_tension_graphs,
)
from shared_core.curve_components import build_offset_curve, find_offset_intersection


class PetgTensionGraphSystemTests(unittest.TestCase):
    def test_graph_variant_inventory_matches_expected_petg_views(self):
        variant_keys = [variant.key for variant in PETG_TENSION_GRAPH_VARIANTS]
        self.assertEqual(
            variant_keys,
            [
                "full_overview",
                "stress_strain_only",
                "elastic_fit_review",
                "offset_yield_review",
                "fracture_review",
            ],
        )

        descriptions = describe_petg_tension_graph_system()
        self.assertEqual(len(descriptions), len(PETG_TENSION_GRAPH_VARIANTS))
        self.assertEqual(PETG_TENSION_OFFSET_STRAIN, 0.002)
        self.assertIn("0.2% offset curve", descriptions[3]["description"].lower())

    def test_renderer_creates_primary_and_component_graph_files(self):
        strain = np.linspace(0.0, 0.012, 7)
        stress = np.array([0.0, 40e6, 78e6, 110e6, 108e6, 96e6, 12e6], dtype=float)
        offset_strain = np.linspace(0.0, 0.012, 200)
        offset_strain, offset_stress = build_offset_curve(
            offset_strain,
            slope=18.5e9,
            offset=PETG_TENSION_OFFSET_STRAIN,
            intercept=0.0,
        )

        tmp_root = Path(__file__).resolve().parents[1] / ".tmp" / "petg_graph_test"
        tmp_root.mkdir(parents=True, exist_ok=True)

        graph_path = tmp_root / "PETG" / "TENSION" / "P1TN40 - 1.html"
        rendered = render_petg_tension_graphs(
            sample_name="P1TN40 - 1",
            display_name="P1TN40 - 1",
            graph_path=graph_path,
            strain=strain,
            stress=stress,
            start_idx=1,
            end_idx=3,
            slope=18.5e9,
            intercept=0.0,
            offset_strain=offset_strain,
            offset_stress=offset_stress,
            yield_strain=0.006,
            yield_stress=110e6,
            yield_method="first_max",
        )

        self.assertTrue(rendered.primary_graph_path.exists())
        self.assertTrue(rendered.manifest_path.exists())
        self.assertEqual(
            set(rendered.component_graph_paths),
            {
                "full_overview",
                "stress_strain_only",
                "elastic_fit_review",
                "offset_yield_review",
                "fracture_review",
            },
        )

        overview_html = rendered.primary_graph_path.read_text(encoding="utf-8")
        self.assertIn("Stress-Strain Curve", overview_html)
        self.assertIn("Linear Fit", overview_html)
        self.assertIn("Offset Curve", overview_html)
        self.assertIn("0.2% Offset Intersection", overview_html)
        self.assertIn("First Max Point", overview_html)
        self.assertIn("Fracture Point", overview_html)

        manifest = json.loads(rendered.manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["offset_strain"], 0.002)
        self.assertEqual(manifest["yield_label"], "First Max Point")

    def test_renderer_can_skip_component_graph_files_for_fast_runs(self):
        strain = np.linspace(0.0, 0.012, 7)
        stress = np.array([0.0, 40e6, 78e6, 110e6, 108e6, 96e6, 12e6], dtype=float)
        offset_strain = np.linspace(0.0, 0.012, 200)
        offset_strain, offset_stress = build_offset_curve(
            offset_strain,
            slope=18.5e9,
            offset=PETG_TENSION_OFFSET_STRAIN,
            intercept=0.0,
        )

        tmp_root = Path(__file__).resolve().parents[1] / ".tmp" / "petg_graph_test_fast"
        tmp_root.mkdir(parents=True, exist_ok=True)

        graph_path = tmp_root / "PETG" / "TENSION" / "P1TN40 - 2.html"
        rendered = render_petg_tension_graphs(
            sample_name="P1TN40 - 2",
            display_name="P1TN40 - 2",
            graph_path=graph_path,
            strain=strain,
            stress=stress,
            start_idx=1,
            end_idx=3,
            slope=18.5e9,
            intercept=0.0,
            offset_strain=offset_strain,
            offset_stress=offset_stress,
            yield_strain=0.006,
            yield_stress=110e6,
            yield_method="first_max",
            write_component_graphs=False,
        )

        self.assertTrue(rendered.primary_graph_path.exists())
        self.assertEqual(rendered.component_graph_paths, {})
        self.assertIsNone(rendered.manifest_path)

    def test_renderer_uses_proof_label_when_petg_yield_method_is_proof(self):
        strain = np.linspace(0.0, 0.012, 7)
        stress = np.array([0.0, 40e6, 78e6, 110e6, 108e6, 96e6, 12e6], dtype=float)
        proof = find_offset_intersection(
            strain,
            stress,
            slope=18.5e9,
            offset=PETG_TENSION_OFFSET_STRAIN,
            intercept=0.0,
            min_strain=float(strain[3]),
        )

        tmp_root = Path(__file__).resolve().parents[1] / ".tmp" / "petg_graph_test_proof"
        tmp_root.mkdir(parents=True, exist_ok=True)

        graph_path = tmp_root / "PETG" / "TENSION" / "P1TN40 - 3.html"
        rendered = render_petg_tension_graphs(
            sample_name="P1TN40 - 3",
            display_name="P1TN40 - 3",
            graph_path=graph_path,
            strain=strain,
            stress=stress,
            start_idx=1,
            end_idx=3,
            slope=18.5e9,
            intercept=0.0,
            offset_strain=proof.offset_strain,
            offset_stress=proof.offset_stress,
            yield_strain=proof.intersection_strain,
            yield_stress=proof.intersection_stress,
            yield_method="proof_stress",
            write_component_graphs=False,
        )

        overview_html = rendered.primary_graph_path.read_text(encoding="utf-8")
        self.assertIn("Yield Point (0.2% Offset)", overview_html)
        self.assertNotIn("0.2% Offset Intersection", overview_html)
        self.assertNotIn("First Max Point", overview_html)


if __name__ == "__main__":
    unittest.main()
