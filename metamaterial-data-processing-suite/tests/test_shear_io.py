from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from openpyxl import Workbook

from shear_core.io import get_values_from_master_file


def build_shifted_shear_dimensions_workbook(path: Path, *, status: str = "P") -> None:
    workbook = Workbook()
    worksheet = workbook.active
    worksheet.title = "Shear_Py"
    worksheet.append(
        [
            "No.",
            "Sample Name",
            "Pre",
            "Unnamed: 3",
            "Unnamed: 4",
            "Unnamed: 5",
            "Avg.WPre",
            "Avg.LPre",
            "P or F",
            "Nbr of sample tested ",
            "Failed sampels",
        ]
    )
    worksheet.append([None, None, "W", None, None, "L", None, None, None, 32, 0])
    worksheet.append([None, None, 1, 2, 3, 1, None, None, None, None, None])
    worksheet.append([1, "P1SH100 - 5", 11.93, 11.92, 11.94, 12.79, 11.93, 12.79, 152.58, status, None])
    workbook.save(path)


class ShearIoTests(unittest.TestCase):
    def test_get_values_from_master_file_supports_shifted_shear_sheet_columns(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workbook_path = Path(temp_dir) / "PET G Sample Dimensions.xlsx"
            build_shifted_shear_dimensions_workbook(workbook_path, status="P")

            gauge_mm, area_mm2 = get_values_from_master_file(
                "P1SH100 - 5",
                dimensions_workbook=workbook_path,
            )

        self.assertEqual(gauge_mm, 12.79)
        self.assertEqual(area_mm2, 152.58)

    def test_get_values_from_master_file_reads_shifted_status_column(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workbook_path = Path(temp_dir) / "PET G Sample Dimensions.xlsx"
            build_shifted_shear_dimensions_workbook(workbook_path, status="F")

            with self.assertRaisesRegex(ValueError, "marked as failed"):
                get_values_from_master_file(
                    "P1SH100 - 5",
                    dimensions_workbook=workbook_path,
                )


if __name__ == "__main__":
    unittest.main()
