from __future__ import annotations

from pathlib import Path

import numpy as np

from shared_core.curve_components import build_offset_curve, clean_curve_data, first_stuck_index


DEFAULT_SHEAR_DIMENSIONS_WORKBOOK = "PET G Sample Dimensions.xlsx"


def default_shear_dimensions_workbook() -> Path:
    here = Path(__file__).resolve().parent.parent
    return here / "resources" / DEFAULT_SHEAR_DIMENSIONS_WORKBOOK


def _first_present_column(row, names):
    for name in names:
        if name in row.index:
            value = row[name]
            if value is not None:
                return value
    return None


def _resolve_status_value(row):
    direct_status = _first_present_column(row, ("P or F",))
    if isinstance(direct_status, str) and direct_status.strip() in {"P", "F"}:
        return direct_status.strip()

    shifted_status = _first_present_column(row, ("Nbr of sample tested ", "Nbr of sample tested"))
    if isinstance(shifted_status, str) and shifted_status.strip() in {"P", "F"}:
        return shifted_status.strip()

    return direct_status


def _resolve_gauge_length_value(row):
    return _first_present_column(row, ("Avg.\nL\nPre", "Avg.LPre"))


def _resolve_area_value(row):
    direct_area = _first_present_column(row, ("Pre Cross sectional area",))
    if direct_area is not None:
        return direct_area

    shifted_area = _first_present_column(row, ("P or F",))
    return shifted_area


def read_shear_xlsx(path: Path, *, dimensions_workbook: Path | None = None):
    import pandas as pd

    df = pd.read_excel(path, engine="calamine")
    gauge_mm, area_mm2 = get_values_from_master_file(
        path.stem,
        dimensions_workbook=dimensions_workbook,
    )

    df.columns = (
        df.columns
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
    )

    lateral_col = next(column for column in df.columns if str(column).startswith("Lateral Strain"))
    axial_col = next(column for column in df.columns if str(column).startswith("Axial Strain"))

    encoder = np.abs(df["Encoder [mm]"].to_numpy(dtype=float)) / gauge_mm
    lvdt = np.abs(df["LVDT [mm]"].to_numpy(dtype=float)) / gauge_mm
    lateral_strain = df[lateral_col].to_numpy(dtype=float) / 1e6
    axial_strain = df[axial_col].to_numpy(dtype=float) / 1e6
    load_N = np.abs(df["Load [kN]"].to_numpy(dtype=float)) * 1000.0

    if area_mm2 is None:
        raise ValueError("area_mm2 must be provided")
    if gauge_mm is None:
        raise ValueError("gauge_mm must be provided")

    area_m2 = float(area_mm2) * 1e-6
    stress_Pa = load_N / area_m2

    return encoder, lvdt, lateral_strain, axial_strain, stress_Pa


def get_values_from_master_file(file_name, *, dimensions_workbook: Path | None = None):
    import pandas as pd

    file = Path(dimensions_workbook) if dimensions_workbook is not None else default_shear_dimensions_workbook()
    df = pd.read_excel(file, sheet_name="Shear_Py", engine="calamine")

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
    status = _resolve_status_value(row)
    if status == "F":
        raise ValueError(f"Sample '{file_name}' is marked as failed (F).")
    if status != "P":
        raise ValueError(
            f"Sample '{file_name}' has invalid status '{status}' in column 'P or F'."
        )

    gauge_mm = _resolve_gauge_length_value(row)
    area_mm2 = _resolve_area_value(row)
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


def trim_beginning_consecutive(axial_strain, lateral_strain, threshold=100 / 1e6, n_consecutive=5):
    axial = np.asarray(axial_strain, dtype=float)
    lateral = np.asarray(lateral_strain, dtype=float)

    if axial.shape != lateral.shape:
        raise ValueError("axial_strain and lateral_strain must have the same size")

    mask = (
        (np.abs(axial) > threshold)
        & (np.abs(lateral) > threshold)
        & (axial < 0)
        & (lateral > 0)
    )

    count = 0
    start_idx = None
    for index, is_valid in enumerate(mask):
        if is_valid:
            count += 1
            if count == n_consecutive:
                start_idx = index - n_consecutive + 1
                break
        else:
            count = 0

    if start_idx is None:
        raise ValueError(f"No sequence of {n_consecutive} consecutive valid points found")

    return axial[start_idx:], lateral[start_idx:], start_idx


def stitch_strain_lvdt(
    enc,
    lateral_strain,
    axial_strain,
    lvdt,
    *,
    tol_strain=0.001,
    window=15,
    min_start=15,
):
    lateral = np.asarray(lateral_strain, dtype=float)
    axial = np.asarray(axial_strain, dtype=float)
    lvdt = np.asarray(lvdt, dtype=float)
    enc = np.asarray(enc, dtype=float)

    n = min(len(lateral), len(axial), len(lvdt), len(enc))
    lateral, axial, lvdt, enc = lateral[:n], axial[:n], lvdt[:n], enc[:n]

    axial, lateral, start_idx = trim_beginning_consecutive(
        axial_strain=axial,
        lateral_strain=lateral,
    )

    lateral_stuck = _first_stuck_index(
        lateral,
        tol_strain=tol_strain,
        window=window,
        min_start=min_start,
    )
    axial_stuck = _first_stuck_index(
        axial,
        tol_strain=tol_strain,
        window=window,
        min_start=min_start,
    )

    candidates = [index for index in (axial_stuck, lateral_stuck) if index is not None]
    if not candidates:
        strain_part = np.abs(axial) + np.abs(lateral)
        return strain_part, start_idx, None, None, start_idx + len(strain_part)

    i1 = min(candidates)
    strain_part = np.abs(axial[:i1]) + np.abs(lateral[:i1])

    slack_strain_to_lvdt = lvdt[start_idx + i1] - strain_part[i1 - 1]
    lvdt_true = lvdt - slack_strain_to_lvdt

    i2 = _first_stuck_index(
        lvdt_true,
        tol_strain=0.001,
        window=window,
        min_start=max(min_start, start_idx + i1 + 1),
    )

    if i2 is None:
        stitched_strain = np.concatenate([
            strain_part,
            lvdt_true[start_idx + i1:],
        ])
        return stitched_strain, start_idx, start_idx + i1, None, start_idx + i1 + len(lvdt_true[start_idx + i1:])

    slack_lvdt_to_enc = enc[i2] - lvdt_true[i2]
    enc_true = enc - slack_lvdt_to_enc
    stitched_strain = np.concatenate([
        strain_part,
        lvdt_true[start_idx + i1:i2],
        enc_true[i2:],
    ])

    return stitched_strain, start_idx, start_idx + i1, i2, i2 + len(enc_true[i2:])


def offset_arrays(strain: np.ndarray, slope: float, offset, intercept):
    return build_offset_curve(strain, slope, offset, intercept)


def clean_data(strain, stress):
    return clean_curve_data(strain, stress)


def trim_drop_after_peak(
    strain,
    stress,
    *,
    min_drop_ratio=0.35,
    min_drop_abs=1.0e6,
):
    strain_array = np.asarray(strain, dtype=float)
    stress_array = np.asarray(stress, dtype=float)

    if len(strain_array) < 3:
        return strain_array, stress_array, None

    mask = np.isfinite(strain_array) & np.isfinite(stress_array)
    strain_array = strain_array[mask]
    stress_array = stress_array[mask]
    if len(stress_array) < 3:
        return strain_array, stress_array, None

    peak_idx = int(np.nanargmax(stress_array))
    for index in range(peak_idx, len(stress_array) - 1):
        current = stress_array[index]
        next_value = stress_array[index + 1]
        drop_abs = current - next_value
        drop_ratio = drop_abs / current if current != 0 else 0.0
        if drop_abs >= min_drop_abs and drop_ratio >= min_drop_ratio:
            cut_idx = index + 1
            return strain_array[:cut_idx], stress_array[:cut_idx], cut_idx

    return strain_array, stress_array, None
