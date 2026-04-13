from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from shared_core.curve_components import build_offset_curve, clean_curve_data, first_stuck_index


def read_tension_xlsx(path: Path, prompt_for_area: bool = True):
    import pandas as pd

    df = pd.read_excel(path, engine="calamine")
    gauge_mm, area_mm2 = get_values_from_master_file(path.stem)

    df.columns = (
        df.columns
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
    )

    encoder = df["Encoder [mm]"].to_numpy(dtype=float) / gauge_mm
    lvdt = df["LVDT [mm]"].to_numpy(dtype=float) / gauge_mm
    extensometer = df["Extensometer [mm]"].to_numpy(dtype=float) / gauge_mm
    load_N = df["Load [kN]"].to_numpy(dtype=float) * 1000.0

    if area_mm2 is None and prompt_for_area:
        while True:
            try:
                raw = input(f"Input the cross sectional Area of {Path(path).name} (mm^2): ")
                area_mm2 = float(raw)
                if area_mm2 <= 0:
                    print("Area must be positive. Try again.")
                    continue
                break
            except ValueError:
                print("Invalid number, try again.")

    if area_mm2 is None:
        raise ValueError("area_mm2 must be provided or prompt_for_area must be True")

    area_m2 = float(area_mm2) * 1e-6
    stress_Pa = load_N / area_m2

    return encoder, lvdt, extensometer, load_N, stress_Pa


@lru_cache(maxsize=1)
def _load_tension_master_rows():
    import pandas as pd

    here = Path(__file__).resolve().parent.parent
    file = here / "resources" / "PET G Sample Dimensions.xlsx"
    df = pd.read_excel(file, sheet_name="Tension_Py", engine="calamine")

    df.columns = (
        df.columns
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
    )
    df["Sample Name"] = df["Sample Name"].astype(str).str.strip()
    df["P or F"] = df["P or F"].astype(str).str.strip()

    rows = {}
    for _, row in df.iterrows():
        sample_name = str(row["Sample Name"]).strip()
        rows[sample_name] = (
            str(row["P or F"]).strip(),
            row["Avg.LPre"],
            row["Pre Cross sectional area"],
        )

    return file.name, rows


def get_values_from_master_file(fileName):
    import pandas as pd

    file_name, rows = _load_tension_master_rows()

    sample_name = str(fileName).strip()
    if sample_name not in rows:
        raise ValueError(
            f"Sample '{fileName}' was not found in column 'Sample Name' of '{file_name}'."
        )

    status, gauge_mm, area_mm2 = rows[sample_name]
    if status == "F":
        raise ValueError(f"Sample '{fileName}' is marked as failed (F).")
    if status != "P":
        raise ValueError(
            f"Sample '{fileName}' has invalid status '{status}' in column 'P or F'."
        )

    if pd.isna(gauge_mm) or pd.isna(area_mm2):
        raise ValueError(
            f"Sample '{fileName}' is missing 'Avg. L Pre' or 'Pre Cross sectional area'."
        )

    return float(gauge_mm), float(area_mm2)


def _first_stuck_index(
    x: np.ndarray,
    *,
    tol_strain: float,
    window: int,
    min_start: int,
):
    return first_stuck_index(
        x,
        tol_strain=tol_strain,
        window=window,
        min_start=min_start,
    )


def stitch_ext_lvdt_encoder(
    extensometer_strain,
    lvdt_strain,
    encoder_strain,
    *,
    tol_strain=0.001,
    window=15,
    min_start=15,
):
    ext = np.asarray(extensometer_strain, dtype=float)
    lvdt = np.asarray(lvdt_strain, dtype=float)
    enc = np.asarray(encoder_strain, dtype=float)

    n = min(len(ext), len(lvdt), len(enc))
    ext, lvdt, enc = ext[:n], lvdt[:n], enc[:n]

    i1 = _first_stuck_index(
        ext,
        tol_strain=tol_strain,
        window=window,
        min_start=min_start,
    )

    if i1 is None:
        return ext.copy(), {
            "used": ["extensometer"],
            "switch_ext_to_lvdt": None,
            "switch_lvdt_to_enc": None,
        }

    slack_ext_to_lvdt = lvdt[i1] - ext[i1]
    lvdt_true = lvdt - slack_ext_to_lvdt

    i2 = _first_stuck_index(
        lvdt_true,
        tol_strain=tol_strain,
        window=window,
        min_start=max(min_start, i1 + 1),
    )

    if i2 is None:
        strain = np.concatenate([
            ext[:i1],
            lvdt_true[i1:],
        ])
        return strain, {
            "used": ["extensometer", "lvdt"],
            "switch_ext_to_lvdt": i1,
            "switch_lvdt_to_enc": None,
        }

    slack_lvdt_to_enc = enc[i2] - lvdt_true[i2]
    enc_true = enc - slack_lvdt_to_enc
    strain = np.concatenate([
        ext[:i1],
        lvdt_true[i1:i2],
        enc_true[i2:],
    ])

    return strain, {
        "used": ["extensometer", "lvdt", "encoder"],
        "switch_ext_to_lvdt": i1,
        "switch_lvdt_to_enc": i2,
    }


def clean_nylon_tension_curve(strain, stress):
    strain_array = np.asarray(strain, dtype=float)
    stress_array = np.asarray(stress, dtype=float)

    mask = np.isfinite(strain_array) & np.isfinite(stress_array)
    strain_array = strain_array[mask]
    stress_array = stress_array[mask]

    if strain_array.size == 0:
        return strain_array, stress_array

    keep_indices = [0]
    max_strain = float(strain_array[0])
    for idx in range(1, len(strain_array)):
        if float(strain_array[idx]) > max_strain:
            keep_indices.append(idx)
            max_strain = float(strain_array[idx])

    keep_indices = np.asarray(keep_indices, dtype=int)
    return strain_array[keep_indices], stress_array[keep_indices]


def offset_arrays(strain: np.ndarray, slope: float, offset, intercept):
    return build_offset_curve(strain, slope, offset, intercept)


def clean_data(strain, stress):
    return clean_curve_data(strain, stress)
