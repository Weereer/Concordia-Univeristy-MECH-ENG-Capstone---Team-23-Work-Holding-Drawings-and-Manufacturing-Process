from __future__ import annotations

from pathlib import Path

import numpy as np
from shared_core.adaptive_fit import (
    ElasticSampleAnalysis,
    cleanup_stale_result_graphs,
    extract_elastic_candidates,
    format_sample_display_name,
    resolve_subgroup_fit_decisions,
)
from shared_core.curve_components import find_offset_intersection

from .constants import (
    BENDING_PRIMARY_R2_MIN,
    BENDING_SLOPE_CV_MAX,
    BENDING_STRESS_GATE_DELTA,
    BENDING_THRESHOLD_STRESS,
    BENDING_WINDOW_STEP,
    BENDING_ZERO_STRAIN_MAX,
    EXPECTED_SAMPLE_SKIP_MARKERS,
)
from .io import (
    read_bending_xlsx,
    stitch_lvdt_encoder,
    transform_data,
    trim_vertical_drop_after_peak,
)
from .results import build_bending_result_row, get_subgroup, save_results_to_xlsx, write_bending_results


def select_bending_fit(
    strain,
    stress,
    *,
    window_min=75,
    window_step=BENDING_WINDOW_STEP,
    r2_min=BENDING_PRIMARY_R2_MIN,
    threshold_stress=BENDING_THRESHOLD_STRESS,
    stress_gate_delta=BENDING_STRESS_GATE_DELTA,
    zero_strain_max=BENDING_ZERO_STRAIN_MAX,
    slope_cv_max=BENDING_SLOPE_CV_MAX,
    fallback_min_points=20,
):
    return extract_elastic_candidates(
        strain,
        stress,
        window_min=window_min,
        window_step=window_step,
        r2_min=r2_min,
        threshold_stress=threshold_stress,
        stress_gate_delta=stress_gate_delta,
        zero_strain_max=zero_strain_max,
        slope_cv_max=slope_cv_max,
        fallback_min_points=fallback_min_points,
    )


def find_linear_region_properties_V3(
    strain,
    stress,
    *,
    window_min=75,
    window_step=BENDING_WINDOW_STEP,
    r2_min=BENDING_PRIMARY_R2_MIN,
    threshold_stress=BENDING_THRESHOLD_STRESS,
    fallback_min_points=20,
):
    _ = fallback_min_points
    search = select_bending_fit(
        strain,
        stress,
        window_min=window_min,
        window_step=window_step,
        r2_min=r2_min,
        threshold_stress=threshold_stress,
        fallback_min_points=fallback_min_points,
    )

    if not search.candidates:
        return np.nan, None, None, np.nan, search.strain, search.stress, np.nan

    best = search.candidates[0]
    return (
        best.slope,
        best.start_idx,
        best.end_idx,
        best.r2,
        search.strain,
        search.stress,
        best.intercept,
    )


def offset_intersection(strain_clean, stress_clean, E, offset, linear_start, linear_end, intercept):
    _ = linear_start
    _ = linear_end
    offset_strain = np.array([], dtype=float)
    offset_stress = np.array([], dtype=float)
    yield_strain = np.nan
    yield_stress = np.nan

    if np.isfinite(E):
        proof = find_offset_intersection(strain_clean, stress_clean, E, offset, intercept)
        offset_strain = proof.offset_strain
        offset_stress = proof.offset_stress
        yield_strain = proof.intersection_strain
        yield_stress = proof.intersection_stress

    ultimate_strength = float(np.nanmax(stress_clean)) if stress_clean.size > 0 else np.nan
    return offset_strain, offset_stress, yield_strain, yield_stress, ultimate_strength


def plot_stress_strain(strain, stress, start_idx, end_idx, slope, offset_strain, offset_stress, offset_intersection_strain, offset_intersection_stress, intercept):
    import matplotlib.pyplot as plt

    for i in range(len(offset_strain)):
        if offset_strain[i] >= offset_intersection_strain:
            offset_strain = offset_strain[: i + 10]
            offset_stress = offset_stress[: i + 10]
            break

    plt.figure(figsize=(8, 6))
    plt.plot(strain, stress, label="Stress-Strain Curve")
    plt.plot(offset_strain, offset_stress, label="Offset Stress-Strain Curve")
    plt.scatter(offset_intersection_strain, offset_intersection_stress, color="red", label="Offset Intersection")
    plt.plot(strain[start_idx:end_idx + 1], stress[start_idx:end_idx + 1], linewidth=3, label="Linear Region")
    x_lin = strain[start_idx:end_idx + 1]
    y_lin = slope * x_lin + intercept
    plt.plot(x_lin, y_lin, linestyle="--", label="Linear Fit")
    plt.xlabel("Strain")
    plt.ylabel("Stress")
    plt.title("Stress-Strain Curve")
    plt.grid(True)
    plt.legend()
    plt.show()


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
    save_path,
    file_name,
):
    import plotly.graph_objects as go

    offset_strain = np.asarray(offset_strain, dtype=float)
    offset_stress = np.asarray(offset_stress, dtype=float)

    if np.isfinite(yield_strain) and np.isfinite(yield_stress):
        for i in range(len(offset_strain)):
            if offset_strain[i] >= yield_strain:
                offset_strain = offset_strain[: i + 10]
                offset_stress = offset_stress[: i + 10]
                break

    has_linear_region = (
        start_idx is not None
        and end_idx is not None
        and 0 <= start_idx <= end_idx < len(strain)
        and np.isfinite(slope)
        and np.isfinite(intercept)
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=strain, y=stress, mode="lines", name="Stress-Strain Curve"))

    if has_linear_region:
        fig.add_trace(go.Scatter(
            x=strain[start_idx:end_idx + 1],
            y=stress[start_idx:end_idx + 1],
            mode="lines",
            line=dict(width=6),
            name="Linear Region",
        ))
        x_lin = strain[start_idx:end_idx + 1]
        y_lin = slope * x_lin + intercept
        fig.add_trace(go.Scatter(
            x=x_lin,
            y=y_lin,
            mode="lines",
            line=dict(dash="dash"),
            name="Linear Fit",
        ))

    if len(offset_strain) > 0 and len(offset_stress) > 0:
        fig.add_trace(go.Scatter(x=offset_strain, y=offset_stress, mode="lines", name="Offset Curve"))

    if np.isfinite(yield_strain) and np.isfinite(yield_stress):
        fig.add_trace(go.Scatter(
            x=[yield_strain],
            y=[yield_stress],
            mode="markers",
            marker=dict(size=10, color="red"),
            name="Yield Point",
        ))

    x_min = np.min(strain)
    x_max = np.max(strain)
    y_min = np.min(stress)
    y_max = np.max(stress)
    x_pad = 0.05 * (x_max - x_min)
    y_pad = 0.05 * (y_max - y_min)

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


def _is_expected_sample_skip(error: Exception) -> bool:
    message = str(error)
    return message.startswith("Sample '") and any(marker in message for marker in EXPECTED_SAMPLE_SKIP_MARKERS)


def _analyze_bending_file(file_name: Path):
    encoder_strain, lvdt_strain, load_N = read_bending_xlsx(file_name)

    disp_mm, _ = stitch_lvdt_encoder(
        lvdt_strain,
        encoder_strain,
        tol_strain=0.001,
        window=12,
        min_start=15,
    )

    strain, stress = transform_data(disp_mm, load_N, file_name)
    strain, stress, _ = trim_vertical_drop_after_peak(
        strain,
        stress,
        strain_tol=8e-4,
        stress_drop_tol=1.0,
    )

    window_min = 100 if "100" in file_name.stem else 75
    search = select_bending_fit(
        strain,
        stress,
        window_min=window_min,
        window_step=BENDING_WINDOW_STEP,
        r2_min=BENDING_PRIMARY_R2_MIN,
        threshold_stress=BENDING_THRESHOLD_STRESS,
        stress_gate_delta=BENDING_STRESS_GATE_DELTA,
        zero_strain_max=BENDING_ZERO_STRAIN_MAX,
        slope_cv_max=BENDING_SLOPE_CV_MAX,
    )

    material = file_name.parts[file_name.parts.index("DATA") + 1]
    return ElasticSampleAnalysis(
        sample_name=file_name.stem,
        file_path=file_name,
        material=material,
        strain=np.asarray(search.strain, dtype=float),
        stress=np.asarray(search.stress, dtype=float),
        candidates=list(search.candidates),
    )


def _finalize_bending_metrics(sample: ElasticSampleAnalysis, graph_root: Path):
    candidate = sample.selected_candidate
    graph_name = format_sample_display_name(sample.sample_name, sample.is_valid, sample.decision_reason)

    if sample.is_valid and candidate is not None:
        offset_strain, offset_stress, yield_strain, yield_stress, ultimate_strength = offset_intersection(
            sample.strain,
            sample.stress,
            candidate.slope,
            0.002,
            candidate.start_idx,
            candidate.end_idx,
            candidate.intercept,
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
        E = np.nan
        start_idx = None
        end_idx = None
        intercept = np.nan
        fit_start_strain = np.nan
        fit_end_strain = np.nan

    graph_path = graph_root / sample.material / "4PT_BENDING" / f"{graph_name}.html"
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
        save_path=graph_path,
        file_name=graph_name,
    )

    return build_bending_result_row(
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


def process_4pt_bending_file(file_name: Path):
    here = Path(__file__).resolve().parent.parent
    sample = _analyze_bending_file(file_name)
    resolve_subgroup_fit_decisions([sample])
    graph_name = format_sample_display_name(sample.sample_name, sample.is_valid, sample.decision_reason)

    excel_path = here / "STATS" / sample.material / f"{sample.material} 4PT_BENDING RESULTS.xlsx"
    row = _finalize_bending_metrics(sample, here / "GRAPHS")
    plot_path = here / "GRAPHS" / sample.material / "4PT_BENDING" / f"{graph_name}.html"
    save_results_to_xlsx(
        sample_name=row[0],
        E=np.nan if row[1] is None or not np.isfinite(row[1]) else float(row[1]) * 1e3,
        yield_strength=np.nan if row[2] is None or not np.isfinite(row[2]) else float(row[2]),
        ultimate_strength=np.nan if row[3] is None or not np.isfinite(row[3]) else float(row[3]),
        output_path=excel_path,
        plot_path=plot_path,
        is_valid=row[5] == "Yes",
        fit_start_strain=row[6],
        fit_end_strain=row[7],
        fit_mode=row[8],
        decision_reason=row[9],
    )
    return sample


def process_4pt_bending_folder(folder: Path):
    here = Path(__file__).resolve().parent.parent
    grouped_samples: dict[str, list[ElasticSampleAnalysis]] = {}

    for file in sorted(folder.glob("*.xlsx")):
        try:
            sample = _analyze_bending_file(file)
        except Exception as e:
            label = "Skipped" if _is_expected_sample_skip(e) else "Failed"
            print(f"{label}: {file.name} -> {e}")
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

    rows = [_finalize_bending_metrics(sample, here / "GRAPHS") for sample in all_samples]
    material = all_samples[0].material
    excel_path = here / "STATS" / material / f"{material} 4PT_BENDING RESULTS.xlsx"
    write_bending_results(excel_path, rows)
    return all_samples
