from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from app_core import runner


class RunnerBatchTests(unittest.TestCase):
    def test_process_shear_folder_uses_shear_batch_system(self):
        folder = Path("DATA") / "PETG" / "PETG SHEAR"

        with patch.object(runner, "process_shear_folder_batch") as batch:
            runner.process_shear_folder(folder)

        batch.assert_called_once_with(folder)

    def test_process_tension_folder_forwards_petg_component_graph_flag(self):
        folder = Path("DATA") / "PETG" / "PETG TENSION"

        with patch.object(runner, "process_tension_folder_batch") as batch:
            runner.process_tension_folder(
                folder,
                petg_component_graphs=False,
            )

        batch.assert_called_once_with(
            folder,
            petg_component_graphs=False,
        )

    def test_main_uses_overview_only_graphs_for_petg_tension_batch(self):
        tension_calls = []

        def record_tension(folder: Path, *, petg_component_graphs: bool = True):
            tension_calls.append((folder.name, petg_component_graphs))

        with patch.object(runner, "process_compression_folder"), patch.object(
            runner,
            "process_4pt_bending_folder",
        ), patch.object(runner, "process_shear_folder"), patch.object(
            runner,
            "process_tension_folder",
            side_effect=record_tension,
        ), patch.object(
            runner,
            "finalize_results_workbook",
        ):
            runner.main()

        self.assertEqual(
            tension_calls,
            [
                ("NYLON TENSION", True),
                ("PETG TENSION", False),
            ],
        )


if __name__ == "__main__":
    unittest.main()
