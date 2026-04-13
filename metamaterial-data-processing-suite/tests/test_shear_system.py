from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from shared_core.adaptive_fit import ElasticSampleAnalysis
from shear_core.system import ShearAnalysisSystem, ShearSystemPaths


def make_sample(sample_name: str, file_path: Path, *, material: str = "PETG") -> ElasticSampleAnalysis:
    sample = ElasticSampleAnalysis(
        sample_name=sample_name,
        file_path=file_path,
        material=material,
        strain=np.array([0.0, 0.01, 0.02], dtype=float),
        stress=np.array([0.0, 1.0, 2.0], dtype=float),
    )
    sample.is_valid = True
    sample.fit_mode = "direct"
    sample.decision_reason = "accepted_clustered"
    return sample


class ShearAnalysisSystemTests(unittest.TestCase):
    def test_process_folder_reports_missing_folder(self):
        paths = ShearSystemPaths(
            project_root=Path("project"),
            stats_root=Path("custom") / "STATS",
            graphs_root=Path("custom") / "GRAPHS",
            dimensions_workbook=Path("custom") / "resources" / "shear-dimensions.xlsx",
        )
        system = ShearAnalysisSystem(paths=paths)
        missing_folder = Path(tempfile.gettempdir()) / "missing-nylon-shear-folder"

        with patch("builtins.print") as print_mock, patch(
            "shear_core.system.analyze_shear_file"
        ) as analyze:
            result = system.process_folder(missing_folder)

        self.assertEqual(result, [])
        analyze.assert_not_called()
        print_mock.assert_called_once_with(
            f"Skipped shear batch: folder not found -> {missing_folder}"
        )

    def test_process_folder_reports_empty_folder(self):
        paths = ShearSystemPaths(
            project_root=Path("project"),
            stats_root=Path("custom") / "STATS",
            graphs_root=Path("custom") / "GRAPHS",
            dimensions_workbook=Path("custom") / "resources" / "shear-dimensions.xlsx",
        )
        system = ShearAnalysisSystem(paths=paths)

        with tempfile.TemporaryDirectory() as temp_dir:
            folder = Path(temp_dir)

            with patch("builtins.print") as print_mock, patch(
                "shear_core.system.analyze_shear_file"
            ) as analyze:
                result = system.process_folder(folder)

        self.assertEqual(result, [])
        analyze.assert_not_called()
        print_mock.assert_called_once_with(
            f"Skipped shear batch: no .xlsx files found in -> {folder}"
        )

    def test_process_file_uses_configured_paths(self):
        paths = ShearSystemPaths(
            project_root=Path("project"),
            stats_root=Path("custom") / "STATS",
            graphs_root=Path("custom") / "GRAPHS",
            dimensions_workbook=Path("custom") / "resources" / "shear-dimensions.xlsx",
        )
        system = ShearAnalysisSystem(paths=paths)
        sample_path = Path("DATA") / "PETG" / "PETG SHEAR" / "P0SH40 - 1.xlsx"
        sample = make_sample("P0SH40 - 1", sample_path)
        row = [
            "P0SH40 - 1",
            1.5,
            2.5,
            3.5,
            '=HYPERLINK("ignored", "Open Graph")',
            "Yes",
            0.001,
            0.002,
            "direct",
            "accepted_clustered",
        ]

        with patch("shear_core.system.analyze_shear_file", return_value=sample) as analyze, patch(
            "shear_core.system.resolve_shear_subgroup_fit_decisions"
        ) as resolve, patch(
            "shear_core.system.finalize_shear_metrics",
            return_value=row,
        ) as finalize, patch("shear_core.system.save_results_to_xlsx") as save, patch.object(
            system,
            "finalize_results_workbook",
        ) as finalize_workbook:
            result = system.process_file(sample_path)

        self.assertIs(result, sample)
        analyze.assert_called_once_with(
            sample_path,
            dimensions_workbook=paths.dimensions_workbook,
        )
        resolve.assert_called_once_with([sample])
        finalize.assert_called_once_with(sample, paths.graphs_root)
        save.assert_called_once_with(
            sample_name="P0SH40 - 1",
            E=1.5e9,
            yield_strength=2.5e6,
            ultimate_strength=3.5e6,
            output_path=paths.stats_root / "PETG" / "PETG SHEAR RESULTS.xlsx",
            plot_path=paths.graphs_root / "PETG" / "SHEAR" / "P0SH40 - 1.html",
            is_valid=True,
            fit_start_strain=0.001,
            fit_end_strain=0.002,
            fit_mode="direct",
            decision_reason="accepted_clustered",
        )
        finalize_workbook.assert_called_once_with("PETG")

    def test_process_folder_batches_by_subgroup_and_writes_once(self):
        paths = ShearSystemPaths(
            project_root=Path("project"),
            stats_root=Path("custom") / "STATS",
            graphs_root=Path("custom") / "GRAPHS",
            dimensions_workbook=Path("custom") / "resources" / "shear-dimensions.xlsx",
        )
        system = ShearAnalysisSystem(paths=paths)

        with tempfile.TemporaryDirectory() as temp_dir:
            folder = Path(temp_dir)
            file_one = folder / "P0SH40 - 1.xlsx"
            file_two = folder / "P0SH40 - 2.xlsx"
            file_three = folder / "P1SH40 - 1.xlsx"
            for file_path in (file_one, file_two, file_three):
                file_path.touch()

            sample_one = make_sample("P0SH40 - 1", file_one)
            sample_two = make_sample("P0SH40 - 2", file_two)
            sample_three = make_sample("P1SH40 - 1", file_three)
            sample_map = {
                file_one.stem: sample_one,
                file_two.stem: sample_two,
                file_three.stem: sample_three,
            }
            row_map = {
                sample_one.sample_name: ["P0SH40 - 1", 1.0, 2.0, 3.0, "", "Yes", 0.1, 0.2, "direct", "accepted_clustered"],
                sample_two.sample_name: ["P0SH40 - 2", 1.1, 2.1, 3.1, "", "Yes", 0.1, 0.2, "direct", "accepted_clustered"],
                sample_three.sample_name: ["P1SH40 - 1", 1.2, 2.2, 3.2, "", "Yes", 0.1, 0.2, "direct", "accepted_clustered"],
            }

            def analyze_side_effect(file_name: Path, *, dimensions_workbook: Path):
                self.assertEqual(dimensions_workbook, paths.dimensions_workbook)
                return sample_map[file_name.stem]

            def finalize_side_effect(sample: ElasticSampleAnalysis, graph_root: Path):
                self.assertEqual(graph_root, paths.graphs_root)
                return row_map[sample.sample_name]

            with patch("shear_core.system.analyze_shear_file", side_effect=analyze_side_effect) as analyze, patch(
                "shear_core.system.resolve_shear_subgroup_fit_decisions"
            ) as resolve, patch(
                "shear_core.system.finalize_shear_metrics",
                side_effect=finalize_side_effect,
            ) as finalize, patch("shear_core.system.write_shear_results") as write, patch.object(
                system,
                "finalize_results_workbook",
            ) as finalize_workbook:
                result = system.process_folder(folder)

        self.assertEqual(len(result), 3)
        self.assertIs(result[0], sample_one)
        self.assertIs(result[1], sample_two)
        self.assertIs(result[2], sample_three)
        self.assertEqual(analyze.call_count, 3)
        self.assertEqual(resolve.call_count, 2)
        first_group = resolve.call_args_list[0].args[0]
        second_group = resolve.call_args_list[1].args[0]
        self.assertEqual(len(first_group), 2)
        self.assertIs(first_group[0], sample_one)
        self.assertIs(first_group[1], sample_two)
        self.assertEqual(len(second_group), 1)
        self.assertIs(second_group[0], sample_three)
        self.assertEqual(finalize.call_count, 3)
        write.assert_called_once_with(
            paths.stats_root / "PETG" / "PETG SHEAR RESULTS.xlsx",
            [
                row_map[sample_one.sample_name],
                row_map[sample_two.sample_name],
                row_map[sample_three.sample_name],
            ],
        )
        finalize_workbook.assert_called_once_with("PETG")


if __name__ == "__main__":
    unittest.main()
