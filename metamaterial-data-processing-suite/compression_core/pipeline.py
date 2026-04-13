from __future__ import annotations

from pathlib import Path

import numpy as np
from shared_core.adaptive_fit import (
    cleanup_stale_result_graphs,
    format_sample_display_name,
)

from .constants import (
    COMPRESSION_CANDIDATE_WINDOW_MAX,
    COMPRESSION_CANDIDATE_WINDOW_MIN,
    COMPRESSION_CANDIDATE_WINDOW_STEP,
    COMPRESSION_PRIMARY_R2_MIN,
    COMPRESSION_PROOF_STRAIN_OFFSET,
    YIELD_METHOD_FIRST_MAX,
    YIELD_METHOD_PROOF,
    YIELD_METHOD_SOFTENING,
)
from .io import read_compression_xlsx, stitch_lvdt_encoder
from .models import CompressionSampleAnalysis
from .results import build_compression_result_row, get_subgroup, save_results_to_xlsx, write_compression_results
from .selection import select_compression_fit
from .subgroup import resolve_subgroup_fit_decisions
from .yielding import determine_compression_yield


EXPECTED_SAMPLE_SKIP_MARKERS = (
    "marked as failed (F)",
    "was not found in column 'Sample Name'",
    "missing 'Avg. L Pre' or 'Pre Cross sectional area'",
)


def _is_expected_sample_skip(error: Exception) -> bool:
    message = str(error)
    return message.startswith("Sample '") and any(marker in message for marker in EXPECTED_SAMPLE_SKIP_MARKERS)

def plot_stress_strain_interactive(
    strain,
    stress,
    start_idx,
    end_idx,
    slope,
    intercept,
    offset_strain,
    offset_stress,
    yield_strain,
    yield_stress,
    yield_method,
    save_path,
    file_name,
):
    import plotly.graph_objects as go

    strain = np.asarray(strain, dtype=float)
    stress = np.asarray(stress, dtype=float)
    offset_strain = np.asarray(offset_strain, dtype=float)
    offset_stress = np.asarray(offset_stress, dtype=float)

    stress_mpa = stress / 1e6
    offset_stress_mpa = offset_stress / 1e6
    yield_stress_mpa = yield_stress / 1e6
    slope_mpa = slope / 1e6
    intercept_mpa = intercept / 1e6

    if (
        yield_method == YIELD_METHOD_PROOF
        and np.isfinite(yield_strain)
        and np.isfinite(yield_stress_mpa)
    ):
        for i in range(len(offset_strain)):
            if offset_strain[i] >= yield_strain:
                offset_strain = offset_strain[: i + 10]
                offset_stress_mpa = offset_stress_mpa[: i + 10]
                break

    has_linear_region = (
        start_idx is not None
        and end_idx is not None
        and 0 <= start_idx <= end_idx < len(strain)
        and np.isfinite(slope_mpa)
        and np.isfinite(intercept_mpa)
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=strain,
        y=stress_mpa,
        mode="lines",
        name="Stress-Strain Curve",
    ))

    if has_linear_region:
        fig.add_trace(go.Scatter(
            x=strain[start_idx : end_idx + 1],
            y=stress_mpa[start_idx : end_idx + 1],
            mode="lines",
            line=dict(width=6),
            name="Linear Region",
        ))

        x_lin = strain[start_idx : end_idx + 1]
        y_lin = slope_mpa * x_lin + intercept_mpa
        fig.add_trace(go.Scatter(
            x=x_lin,
            y=y_lin,
            mode="lines",
            line=dict(dash="dash"),
            name="Linear Fit",
        ))

    if len(offset_strain) > 0 and len(offset_stress_mpa) > 0:
        fig.add_trace(go.Scatter(
            x=offset_strain,
            y=offset_stress_mpa,
            mode="lines",
            line=dict(dash="dash", width=2, color="mediumpurple"),
            opacity=0.9,
            name="Offset Curve",
        ))

    if np.isfinite(yield_strain) and np.isfinite(yield_stress_mpa):
        marker_name = "Yield Point"
        if yield_method == YIELD_METHOD_SOFTENING:
            marker_name = "Yield Point (Knee)"
        elif yield_method == YIELD_METHOD_FIRST_MAX:
            marker_name = "Yield Point (First Maximum)"
        elif yield_method == YIELD_METHOD_PROOF:
            marker_name = "Yield Point (Proof Stress)"
        fig.add_trace(go.Scatter(
            x=[yield_strain],
            y=[yield_stress_mpa],
            mode="markers",
            marker=dict(size=10, color="red"),
            name=marker_name,
        ))

    x_min = float(np.min(strain)) if strain.size > 0 else 0.0
    x_max = float(np.max(strain)) if strain.size > 0 else 1.0
    y_min = float(np.min(stress_mpa)) if stress_mpa.size > 0 else 0.0
    y_max = float(np.max(stress_mpa)) if stress_mpa.size > 0 else 1.0

    x_pad = 0.05 * max(x_max - x_min, 1e-6)
    y_pad = 0.05 * max(y_max - y_min, 1e-6)

    fig.update_layout(
        title="Stress-Strain Curve of " + file_name,
        title_x=0.5,
        xaxis_title="Strain",
        yaxis_title="Stress (MPa)",
        template="plotly_white",
        xaxis=dict(range=[x_min - x_pad, x_max + x_pad]),
        yaxis=dict(range=[y_min - y_pad, y_max + y_pad]),
    )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(save_path)
    return save_path


def _analyze_compression_file(file_name: Path):
    encoder_strain, lvdt_strain, _, stress_Pa, _ = read_compression_xlsx(
        file_name,
        prompt_for_area=True,
        prompt_for_gauge=True,
    )

    strain, _ = stitch_lvdt_encoder(
        lvdt_strain,
        encoder_strain,
        tol_strain=0.001,
        window=15,
        min_start=15,
    )

    fit = select_compression_fit(
        strain,
        stress_Pa,
        window_min=COMPRESSION_CANDIDATE_WINDOW_MIN,
        primary_window_step=COMPRESSION_CANDIDATE_WINDOW_STEP,
        r2_min=COMPRESSION_PRIMARY_R2_MIN,
        threshold_stress=0.0,
        window_max=COMPRESSION_CANDIDATE_WINDOW_MAX,
    )

    material = file_name.parts[file_name.parts.index("DATA") + 1]
    return CompressionSampleAnalysis(
        sample_name=file_name.stem,
        file_path=file_name,
        material=material,
        strain=np.asarray(fit["strain"], dtype=float),
        stress=np.asarray(fit["stress"], dtype=float),
        candidates=list(fit["candidates"]),
        yield_cutoff_idx=fit["yield_cutoff_idx"],
        geometric_instability=fit["geometric_instability"],
    )


def _finalize_sample_metrics(sample: CompressionSampleAnalysis, graph_root: Path):
    candidate = sample.selected_candidate
    graph_name = format_sample_display_name(sample.sample_name, sample.is_valid, sample.decision_reason)

    if sample.is_valid and candidate is not None:
        offset_strain, offset_stress, yield_strain, yield_stress, ultimate_strength, yield_method = determine_compression_yield(
            sample.strain,
            sample.stress,
            candidate.slope,
            COMPRESSION_PROOF_STRAIN_OFFSET,
            candidate.start_idx,
            candidate.end_idx,
            candidate.intercept,
            yield_cutoff_idx=sample.yield_cutoff_idx,
        )
        E = candidate.slope
        start_idx = candidate.start_idx
        end_idx = candidate.end_idx
        intercept = candidate.intercept
        fit_start_strain = candidate.start_strain
        fit_end_strain = candidate.end_strain
    else:
        offset_strain = np.array([], dtype=float)
        offset_stress = np.array([], dtype=float)
        yield_strain = np.nan
        yield_stress = np.nan
        ultimate_strength = float(np.nanmax(sample.stress)) if sample.stress.size > 0 else np.nan
        yield_method = ""
        E = np.nan
        start_idx = None
        end_idx = None
        intercept = np.nan
        fit_start_strain = np.nan
        fit_end_strain = np.nan

    graph_path = graph_root / sample.material / "COMPRESSION" / f"{graph_name}.html"
    cleanup_stale_result_graphs(graph_path, sample.sample_name)
    graph_link = plot_stress_strain_interactive(
        sample.strain,
        sample.stress,
        start_idx,
        end_idx,
        E,
        intercept,
        offset_strain,
        offset_stress,
        yield_strain,
        yield_stress,
        yield_method,
        save_path=graph_path,
        file_name=graph_name,
    )

    return build_compression_result_row(
        sample_name=sample.sample_name,
        E=E,
        yield_strength=yield_stress,
        ultimate_strength=ultimate_strength,
        plot_path=graph_link,
        is_valid=sample.is_valid,
        fit_start_strain=fit_start_strain,
        fit_end_strain=fit_end_strain,
        fit_mode=sample.fit_mode,
        decision_reason=sample.decision_reason,
    )


def process_compression_file(file_name: Path):
    here = Path(__file__).resolve().parent.parent
    sample = _analyze_compression_file(file_name)
    resolve_subgroup_fit_decisions([sample])
    graph_name = format_sample_display_name(sample.sample_name, sample.is_valid, sample.decision_reason)

    excel_path = here / "STATS" / sample.material / f"{sample.material} COMPRESSION RESULTS.xlsx"
    row = _finalize_sample_metrics(sample, here / "GRAPHS")
    plot_path = here / "GRAPHS" / sample.material / "COMPRESSION" / f"{graph_name}.html"
    save_results_to_xlsx(
        sample_name=row[0],
        E=np.nan if row[1] is None or not np.isfinite(row[1]) else float(row[1]) * 1e9,
        yield_strength=np.nan if row[2] is None or not np.isfinite(row[2]) else float(row[2]) * 1e6,
        ultimate_strength=np.nan if row[3] is None or not np.isfinite(row[3]) else float(row[3]) * 1e6,
        output_path=excel_path,
        plot_path=plot_path,
        is_valid=row[5] == "Yes",
        fit_start_strain=row[6],
        fit_end_strain=row[7],
        fit_mode=row[8],
        decision_reason=row[9],
    )
    return sample


def process_compression_folder(folder: Path):
    here = Path(__file__).resolve().parent.parent
    grouped_samples: dict[str, list[CompressionSampleAnalysis]] = {}

    for file in sorted(folder.glob("*.xlsx")):
        try:
            sample = _analyze_compression_file(file)
        except Exception as error:
            label = "Skipped" if _is_expected_sample_skip(error) else "Failed"
            print(f"{label}: {file.name} -> {error}")
            continue
        subgroup = get_subgroup(sample.sample_name)
        grouped_samples.setdefault(subgroup, []).append(sample)

    all_samples = []
    for subgroup in sorted(grouped_samples):
        subgroup_samples = grouped_samples[subgroup]
        resolve_subgroup_fit_decisions(subgroup_samples)
        all_samples.extend(subgroup_samples)

    if not all_samples:
        return []

    rows = [_finalize_sample_metrics(sample, here / "GRAPHS") for sample in all_samples]
    material = all_samples[0].material
    excel_path = here / "STATS" / material / f"{material} COMPRESSION RESULTS.xlsx"
    write_compression_results(excel_path, rows)
    return all_samples
