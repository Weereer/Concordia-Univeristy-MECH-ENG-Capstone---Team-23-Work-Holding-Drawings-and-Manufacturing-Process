from __future__ import annotations

from pathlib import Path

import numpy as np

from shared_core.curve_components import build_offset_curve, clean_curve_data, first_stuck_index


def read_compression_xlsx(path: Path, prompt_for_area: bool = True, prompt_for_gauge: bool = True):
    import pandas as pd

    df = pd.read_excel(path, engine="calamine")
    gauge_mm, area_mm2 = get_values_from_master_file(path.stem)

    df.columns = (
        df.columns
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
    )

    if gauge_mm is None and prompt_for_gauge:
        while True:
            try:
                raw = input(f"Input the gauge length of {Path(path).name} (mm): ")
                gauge_mm = float(raw)
                if gauge_mm <= 0:
                    print("Gauge length must be positive. Try again.")
                    continue
                break
            except ValueError:
                print("Invalid number, try again.")

    if gauge_mm is None:
        raise ValueError("gauge_mm must be provided or prompt_for_gauge must be True")

    gauge = float(gauge_mm)
    encoder = np.abs(df["Encoder [mm]"].to_numpy(dtype=float)) / gauge
    lvdt = np.abs(df["LVDT [mm]"].to_numpy(dtype=float)) / gauge
    load_N = np.abs(df["Load [kN]"].to_numpy(dtype=float)) * 1000.0

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
    return encoder, lvdt, load_N, stress_Pa, float(area_mm2)


def get_values_from_master_file(file_name):
    import pandas as pd

    here = Path(__file__).resolve().parent.parent
    file = here / "resources" / "PET G Sample Dimensions.xlsx"
    df = pd.read_excel(file, sheet_name="Compression_Py", engine="calamine")

    df.columns = (
        df.columns
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
    )
    df["Sample Name"] = df["Sample Name"].astype(str).str.strip()
    df["P or F"] = df["P or F"].astype(str).str.strip()

    match = df[df["Sample Name"] == str(file_name).strip()]
    if match.empty:
        raise ValueError(
            f"Sample '{file_name}' was not found in column 'Sample Name' of '{file.name}'."
        )

    row = match.iloc[0]
    status = row["P or F"]
    if status == "F":
        raise ValueError(f"Sample '{file_name}' is marked as failed (F).")
    if status != "P":
        raise ValueError(
            f"Sample '{file_name}' has invalid status '{status}' in column 'P or F'."
        )

    gauge_mm = row["Avg.LPre"]
    area_mm2 = row["Pre Cross sectional area"]
    if pd.isna(gauge_mm) or pd.isna(area_mm2):
        raise ValueError(
            f"Sample '{file_name}' is missing 'Avg. L Pre' or 'Pre Cross sectional area'."
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


def stitch_lvdt_encoder(
    lvdt_strain,
    encoder_strain,
    *,
    tol_strain=0.001,
    window=15,
    min_start=15,
):
    lvdt = np.asarray(lvdt_strain, dtype=float)
    enc = np.asarray(encoder_strain, dtype=float)

    n = min(len(lvdt), len(enc))
    lvdt = lvdt[:n]
    enc = enc[:n]

    i_switch = _first_stuck_index(
        lvdt,
        tol_strain=tol_strain,
        window=window,
        min_start=min_start,
    )

    if i_switch is None:
        return lvdt.copy(), {
            "used": ["lvdt"],
            "switch_lvdt_to_encoder": None,
        }

    slack_lvdt_to_enc = enc[i_switch] - lvdt[i_switch]
    enc_true = enc - slack_lvdt_to_enc
    strain = np.concatenate([lvdt[:i_switch], enc_true[i_switch:]])

    return strain, {
        "used": ["lvdt", "encoder"],
        "switch_lvdt_to_encoder": i_switch,
    }


def offset_arrays(strain: np.ndarray, slope: float, offset, intercept):
    return build_offset_curve(strain, slope, offset, intercept)


def clean_data(strain, stress):
    return clean_curve_data(strain, stress)
