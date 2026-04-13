from __future__ import annotations

import argparse
from pathlib import Path

from tension_core import process_tension_folder


def _default_petg_tension_folder() -> Path:
    return Path(__file__).resolve().parent.parent / "DATA" / "PETG" / "PETG TENSION"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run only the PETG tension graph/results pipeline.",
    )
    parser.add_argument(
        "folder",
        nargs="?",
        default=str(_default_petg_tension_folder()),
        help="PETG tension input folder. Defaults to DATA/PETG/PETG TENSION.",
    )
    parser.add_argument(
        "--overview-only",
        action="store_true",
        help="Write only the main PETG overview graph and skip component graph files.",
    )
    args = parser.parse_args(argv)

    folder = Path(args.folder).expanduser().resolve()
    if not folder.exists():
        raise SystemExit(f"PETG tension folder was not found: {folder}")
    if not folder.is_dir():
        raise SystemExit(f"PETG tension path must be a folder: {folder}")

    process_tension_folder(
        folder,
        petg_component_graphs=not args.overview_only,
    )
    print(f"PETG tension processing complete: {folder}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
