from __future__ import annotations

from pathlib import Path

import numpy as np

from shared_core.curve_components import build_offset_curve, clean_curve_data, first_stuck_index


def read_bending_xlsx(path: Path):
    import pandas as pd

    df = pd.read_excel(path, engine="calamine")
    df.columns = (
        df.columns
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
    )

    encoder = df["Encoder [mm]"].to_numpy(dtype=float)
    lvdt = df["LVDT [mm]"].to_numpy(dtype=float)
    load_N = df["Load [kN]"].to_numpy(dtype=float) * 1000.0

    clean_idx = trim_to_first_valid_run(lvdt, n=7)
    return encoder[clean_idx:], lvdt[clean_idx:], load_N[clean_idx:]


def trim_to_first_valid_run(arr, n=7):
    for i in range(len(arr) - n + 1):
        if np.all(arr[i:i + n] != 0):
            return i
    return 0


def transform_data(mix_data: np.ndarray, load_N: np.ndarray, path: Path):
    support_span = 158
    beam_depth, beam_width = get_values_from_master_file(path.stem)

    strain = mix_data * (beam_depth / (0.23 * support_span**2))
    stress = load_N * ((3 * support_span) / (4 * beam_width * beam_depth**2))
    return strain, stress


def trim_vertical_drop_after_peak(
    strain,
    stress,
    *,
    strain_tol=6e-4,
    stress_drop_tol=1.0,
):
    strain = np.asarray(strain, dtype=float)
    stress = np.asarray(stress, dtype=float)

    if len(strain) < 3:
        return strain, stress, None

    peak_idx = int(np.nanargmax(stress))
    dstrain = np.diff(strain)
    dstress = np.diff(stress)

    for i in range(peak_idx, len(dstrain)):
        small_strain_change = abs(dstrain[i]) <= strain_tol
        big_stress_drop = dstress[i] <= -stress_drop_tol

        if small_strain_change and big_stress_drop:
            cut_idx = i + 1
            return strain[:cut_idx], stress[:cut_idx], cut_idx

    return strain, stress, None


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
    strain = np.concatenate([
        lvdt[:i_switch],
        enc_true[i_switch:],
    ])

    return strain, {
        "used": ["lvdt", "encoder"],
        "switch_lvdt_to_encoder": i_switch,
    }


def get_values_from_master_file(fileName):
    import pandas as pd

    here = Path(__file__).resolve().parent.parent
    file = here / "resources" / "PET G Sample Dimensions.xlsx"
    df = pd.read_excel(file, sheet_name="Bending_Py", engine="calamine")

    df.columns = (
        df.columns
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
    )
    df["Sample Name"] = df["Sample Name"].astype(str).str.strip()
    df["P or F"] = df["P or F"].astype(str).str.strip()

    match = df[df["Sample Name"] == str(fileName).strip()]
    if match.empty:
        raise ValueError(
            f"Sample '{fileName}' was not found in column 'Sample Name' of '{file.name}'."
        )

    row = match.iloc[0]
    status = row["P or F"]
    if status == "F":
        raise ValueError(f"Sample '{fileName}' is marked as failed (F).")
    if status != "P":
        raise ValueError(
            f"Sample '{fileName}' has invalid status '{status}' in column 'P or F'."
        )

    beam_width = row["Avg.WPre"]
    beam_depth = row["Avg.LPre"]
    if pd.isna(beam_width) or pd.isna(beam_depth):
        raise ValueError(
            f"Sample '{fileName}' is missing 'Avg. L Pre' or 'Pre Cross sectional area'."
        )

    return float(beam_width), float(beam_depth)


def offset_arrays(strain: np.ndarray, slope: float, offset, intercept):
    return build_offset_curve(strain, slope, offset, intercept)


def clean_data(strain, stress):
    return clean_curve_data(strain, stress)
