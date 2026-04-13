from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from tkinter import Tk, filedialog

from bending_core import process_4pt_bending_folder as process_4pt_bending_folder_batch
from compression_core import process_compression_folder as process_compression_folder_batch
from reporting_core import finalize_results_workbook
from shear_core import process_shear_folder as process_shear_folder_batch
from tension_core import process_tension_folder as process_tension_folder_batch


MATERIAL_MAP = {"P": "PETG", "N": "Nylon"}
ORIENTATION_MAP = {"0": "Short axis aligned", "1": "Long axis aligned", "2": "Long axis flipped"}
TEST_MAP = {"TN": "Tension", "BE": "Bending", "CP": "Compression", "SH": "Shear"}

NAME_RE = re.compile(
    r"^(?P<mat>[PN])(?P<ori>[0-2])(?P<test>TN|BE|CP|SH)(?P<infill>\d+) - (?P<sample>\d+)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SampleInfo:
    file: Path
    material: str
    orientation: str
    test_type: str
    infill: int
    sample: int


def parse_sample_name(path: Path) -> SampleInfo:
    match = NAME_RE.match(path.stem)
    if not match:
        raise ValueError(f"Bad filename '{path.name}'. Expected like P1BE100-3.csv")

    mat_code = match.group("mat").upper()
    ori_code = match.group("ori")
    test_code = match.group("test").upper()

    return SampleInfo(
        file=path,
        material=MATERIAL_MAP.get(mat_code, f"Unknown({mat_code})"),
        orientation=ORIENTATION_MAP.get(ori_code, f"Unknown({ori_code})"),
        test_type=TEST_MAP.get(test_code, f"Unknown({test_code})"),
        infill=int(match.group("infill")),
        sample=int(match.group("sample")),
    )


def select_csv_files() -> list[Path]:
    root = Tk()
    root.withdraw()
    files = filedialog.askopenfilenames(
        title="Select machine CSV files (e.g., P1BE100-1.xlsx...)",
        filetypes=[("xlsx files", "*.xlsx")],
    )
    root.destroy()
    return [Path(f) for f in files]


def process_4pt_bending_folder(folder: Path):
    process_4pt_bending_folder_batch(folder)


def process_compression_folder(folder: Path):
    process_compression_folder_batch(folder)


def process_tension_folder(folder: Path, *, petg_component_graphs: bool = True):
    process_tension_folder_batch(
        folder,
        petg_component_graphs=petg_component_graphs,
    )


def process_shear_folder(folder: Path):
    process_shear_folder_batch(folder)


def main() -> None:
    here = Path(__file__).resolve().parent.parent
    data_folder = here / "DATA"

    petg_4pt_bending_folder = data_folder / "PETG" / "PETG 4PT_BENDING"
    petg_shear_folder = data_folder / "PETG" / "PETG SHEAR"
    petg_compression_folder = data_folder / "PETG" / "PETG COMPRESSION"
    petg_tension_folder = data_folder / "PETG" / "PETG TENSION"

    nylon_compression_folder = data_folder / "NYLON" / "NYLON COMPRESSION"
    nylon_tension_folder = data_folder / "NYLON" / "NYLON TENSION"
    nylon_shear_folder = data_folder / "NYLON" / "NYLON SHEAR"

    print(f"Starting compression batch: {nylon_compression_folder.name}")
    process_compression_folder(nylon_compression_folder)
    print(f"Starting compression batch: {petg_compression_folder.name}")
    process_compression_folder(petg_compression_folder)

    print(f"Starting tension batch: {nylon_tension_folder.name}")
    process_tension_folder(nylon_tension_folder)
    print(f"Starting tension batch: {petg_tension_folder.name} (overview-only PETG graphs)")
    process_tension_folder(
        petg_tension_folder,
        petg_component_graphs=False,
    )

    print(f"Starting bending batch: {petg_4pt_bending_folder.name}")
    process_4pt_bending_folder(petg_4pt_bending_folder)

    print(f"Starting shear batch: {petg_shear_folder.name}")
    process_shear_folder(petg_shear_folder)
    print(f"Starting shear batch: {nylon_shear_folder.name}")
    process_shear_folder(nylon_shear_folder)

    stats_dir = here / "STATS"
    result_files = [
        stats_dir / "NYLON" / "NYLON COMPRESSION RESULTS.xlsx",
        stats_dir / "PETG" / "PETG COMPRESSION RESULTS.xlsx",
        stats_dir / "NYLON" / "NYLON TENSION RESULTS.xlsx",
        stats_dir / "PETG" / "PETG TENSION RESULTS.xlsx",
        stats_dir / "PETG" / "PETG 4PT_BENDING RESULTS.xlsx",
        stats_dir / "PETG" / "PETG SHEAR RESULTS.xlsx",
        stats_dir / "NYLON" / "NYLON SHEAR RESULTS.xlsx",
    ]

    for workbook in result_files:
        if workbook.exists():
            print(f"Finalizing workbook: {workbook.name}")
            finalize_results_workbook(workbook)


if __name__ == "__main__":
    main()
