from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from shared_core.adaptive_fit import canonical_sample_name
from shared_core.curve_components import find_curve_intersection

from .components import (
    PETG_TENSION_GRAPH_VARIANTS,
    PETG_TENSION_OFFSET_STRAIN,
    describe_petg_tension_graph_system,
)


_OFFSET_DISPLAY_PADDING_POINTS = 10
_WINDUP_DISPLAY_START_STRAIN_MIN = 0.01
_LINE_DISPLAY_BACKTRACK_RESIDUAL_FRACTION = 0.10
_LINE_DISPLAY_BACKTRACK_RESIDUAL_MIN = 0.5e6
_LINE_DISPLAY_BACKTRACK_MAX_STRAIN_DELTA = 0.003
_LINE_DISPLAY_BACKTRACK_MIN_START_STRESS_RATIO = 0.82
_FAILURE_STEP_DROP_FRACTION = 0.25
_FAILURE_NEAR_ZERO_TAIL_FRACTION = 0.10
_FAILURE_RECOVERY_RATIO_MIN = 0.75

TENSION_YIELD_METHOD_PROOF = "proof_stress"
TENSION_YIELD_METHOD_FIRST_MAX = "first_max"


@dataclass(frozen=True)
class PetgTensionRenderedGraphs:
    primary_graph_path: Path
    component_graph_paths: dict[str, Path]
    manifest_path: Path | None


def _yield_label_for_petg(yield_method: str) -> str:
    if yield_method == TENSION_YIELD_METHOD_PROOF:
        return "Yield Point (0.2% Offset)"
    if yield_method == TENSION_YIELD_METHOD_FIRST_MAX:
        return "First Max Point"
    return "Yield Point"


def _proof_marker_label_for_petg(yield_method: str) -> str:
    if yield_method == TENSION_YIELD_METHOD_PROOF:
        return "Yield Point (0.2% Offset)"
    return "0.2% Offset Intersection"


def _proof_search_start_strain(strain, start_idx, end_idx):
    strain = np.asarray(strain, dtype=float)
    if strain.size == 0:
        return None
    if end_idx is not None and 0 <= int(end_idx) < len(strain):
        return float(strain[int(end_idx)])
    if start_idx is not None and 0 <= int(start_idx) < len(strain):
        return float(strain[int(start_idx)])
    return None


def _has_tension_recovery_after_index(
    stress_clean: np.ndarray,
    start_idx: int,
    *,
    reference_stress: float,
):
    stress_clean = np.asarray(stress_clean, dtype=float)
    if stress_clean.size == 0:
        return False

    if not np.isfinite(reference_stress) or reference_stress <= 0.0:
        return False

    start = max(int(start_idx), 0)
    if start >= stress_clean.size:
        return False

    tail = stress_clean[start:]
    finite_tail = tail[np.isfinite(tail)]
    if finite_tail.size == 0:
        return False

    recovered_peak = float(np.nanmax(finite_tail))
    return recovered_peak >= _FAILURE_RECOVERY_RATIO_MIN * float(reference_stress)


def find_tension_analysis_end_idx(stress_clean: np.ndarray):
    stress_clean = np.asarray(stress_clean, dtype=float)
    if stress_clean.size == 0:
        return None
    if stress_clean.size == 1:
        return 0

    ultimate_idx = int(np.nanargmax(stress_clean))
    ultimate_strength = float(stress_clean[ultimate_idx])
    if not np.isfinite(ultimate_strength) or ultimate_strength <= 0.0:
        return len(stress_clean) - 1

    catastrophic_step_drop = -_FAILURE_STEP_DROP_FRACTION * ultimate_strength
    near_zero_tail_limit = _FAILURE_NEAR_ZERO_TAIL_FRACTION * ultimate_strength
    tail_max = np.maximum.accumulate(stress_clean[::-1])[::-1]

    for idx in range(ultimate_idx + 1, len(stress_clean)):
        if float(stress_clean[idx] - stress_clean[idx - 1]) <= catastrophic_step_drop:
            if _has_tension_recovery_after_index(
                stress_clean,
                idx,
                reference_stress=float(stress_clean[idx - 1]),
            ):
                continue
            return max(idx - 1, ultimate_idx)
        if float(tail_max[idx]) <= near_zero_tail_limit:
            return max(idx - 1, ultimate_idx)

    return len(stress_clean) - 1


def _finite_plot_bounds(*series, fallback: tuple[float, float]):
    finite_values = []
    for values in series:
        array = np.asarray(values, dtype=float).reshape(-1)
        if array.size == 0:
            continue
        finite = array[np.isfinite(array)]
        if finite.size > 0:
            finite_values.append(finite)

    if not finite_values:
        return fallback

    combined = np.concatenate(finite_values)
    return float(np.min(combined)), float(np.max(combined))


def _format_offset_curve_for_marker_display(
    offset_strain,
    offset_stress,
    marker_strain,
    marker_stress,
    *,
    min_display_stress: float | None = None,
    min_display_strain: float | None = None,
):
    offset_strain = np.asarray(offset_strain, dtype=float)
    offset_stress = np.asarray(offset_stress, dtype=float)

    if (
        offset_strain.size == 0
        or offset_stress.size == 0
        or not np.isfinite(marker_strain)
        or not np.isfinite(marker_stress)
    ):
        return offset_strain, offset_stress

    display_start = 0
    if min_display_stress is not None and np.isfinite(min_display_stress):
        visible_start = next(
            (index for index, value in enumerate(offset_stress) if value >= float(min_display_stress)),
            None,
        )
        if visible_start is not None:
            display_start = max(visible_start - _OFFSET_DISPLAY_PADDING_POINTS, 0)
    if min_display_strain is not None and np.isfinite(min_display_strain):
        visible_start = next(
            (index for index, value in enumerate(offset_strain) if value >= float(min_display_strain)),
            None,
        )
        if visible_start is not None:
            display_start = max(display_start, visible_start)

    crossing_index = next(
        (index for index, value in enumerate(offset_strain) if value >= float(marker_strain)),
        None,
    )
    if crossing_index is None:
        return offset_strain[display_start:], offset_stress[display_start:]

    display_end = min(crossing_index + _OFFSET_DISPLAY_PADDING_POINTS, offset_strain.size)
    if display_end <= display_start:
        return offset_strain, offset_stress
    return offset_strain[display_start:display_end], offset_stress[display_start:display_end]


def _tension_display_start_idx(strain, stress) -> int:
    strain = np.asarray(strain, dtype=float)
    stress = np.asarray(stress, dtype=float)
    if strain.size == 0 or stress.size == 0:
        return 0

    threshold_hits = np.flatnonzero(np.isfinite(stress) & (stress >= 1.24e6))
    if threshold_hits.size == 0:
        return 0

    first_loaded_idx = int(threshold_hits[0])
    if first_loaded_idx <= _OFFSET_DISPLAY_PADDING_POINTS:
        return 0

    loaded_strain = float(strain[min(first_loaded_idx, strain.size - 1)])
    if not np.isfinite(loaded_strain) or loaded_strain < _WINDUP_DISPLAY_START_STRAIN_MIN:
        return 0

    return max(first_loaded_idx - _OFFSET_DISPLAY_PADDING_POINTS, 0)


def _line_display_start_idx(strain, stress, start_idx, end_idx, slope, intercept) -> int:
    strain = np.asarray(strain, dtype=float)
    stress = np.asarray(stress, dtype=float)
    if strain.size == 0 or stress.size == 0:
        return 0
    if start_idx is None or end_idx is None:
        return 0
    if not np.isfinite(slope) or not np.isfinite(intercept):
        return max(int(start_idx), 0)

    start = max(int(start_idx), 0)
    end = min(int(end_idx), len(stress) - 1)
    if end < start:
        return start

    fit_stress = stress[start : end + 1]
    finite_fit_stress = fit_stress[np.isfinite(fit_stress)]
    if finite_fit_stress.size == 0:
        return start

    fit_start_stress = float(stress[start])
    min_start_stress = fit_start_stress * _LINE_DISPLAY_BACKTRACK_MIN_START_STRESS_RATIO
    min_display_strain = float(strain[start]) - _LINE_DISPLAY_BACKTRACK_MAX_STRAIN_DELTA
    fit_stress_span = float(np.max(finite_fit_stress) - np.min(finite_fit_stress))
    residual_tol = max(
        float(fit_stress_span) * _LINE_DISPLAY_BACKTRACK_RESIDUAL_FRACTION,
        _LINE_DISPLAY_BACKTRACK_RESIDUAL_MIN,
    )

    display_start = start
    while display_start > 0:
        prev_idx = display_start - 1
        if float(strain[prev_idx]) < min_display_strain:
            break
        predicted_stress = float(slope) * float(strain[prev_idx]) + float(intercept)
        actual_stress = float(stress[prev_idx])
        if not np.isfinite(predicted_stress) or not np.isfinite(actual_stress):
            break
        if np.isfinite(fit_start_stress) and actual_stress < min_start_stress:
            break
        if abs(actual_stress - predicted_stress) > residual_tol:
            break
        display_start = prev_idx

    return display_start


def _fracture_point(strain, stress):
    strain = np.asarray(strain, dtype=float)
    stress = np.asarray(stress, dtype=float)
    if strain.size == 0 or stress.size == 0:
        return np.nan, np.nan

    analysis_end_idx = find_tension_analysis_end_idx(stress)
    if analysis_end_idx is None:
        analysis_end_idx = min(len(strain), len(stress)) - 1
    analysis_end_idx = min(int(analysis_end_idx), len(strain) - 1, len(stress) - 1)
    return float(strain[analysis_end_idx]), float(stress[analysis_end_idx])


def _proof_point(strain, stress, offset_strain, offset_stress, *, min_strain=None):
    strain = np.asarray(strain, dtype=float)
    stress = np.asarray(stress, dtype=float)
    offset_strain = np.asarray(offset_strain, dtype=float)
    offset_stress = np.asarray(offset_stress, dtype=float)

    if (
        strain.size == 0
        or stress.size == 0
        or offset_strain.size == 0
        or offset_stress.size == 0
    ):
        return np.nan, np.nan

    proof_strain, proof_stress = find_curve_intersection(
        strain,
        stress,
        offset_strain,
        offset_stress,
        min_strain=min_strain,
    )
    return float(proof_strain), float(proof_stress)


def _prepare_petg_plot_data(
    *,
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
    fracture_strain,
    fracture_stress,
):
    strain = np.asarray(strain, dtype=float)
    stress = np.asarray(stress, dtype=float)
    offset_strain = np.asarray(offset_strain, dtype=float)
    offset_stress = np.asarray(offset_stress, dtype=float)

    analysis_end_idx = find_tension_analysis_end_idx(stress)
    if analysis_end_idx is not None and strain.size > 0 and stress.size > 0:
        analysis_end_idx = min(int(analysis_end_idx), len(strain) - 1, len(stress) - 1)
        strain = strain[: analysis_end_idx + 1]
        stress = stress[: analysis_end_idx + 1]
        if offset_strain.size > 0 and offset_stress.size > 0:
            offset_mask = offset_strain <= float(strain[-1])
            offset_strain = offset_strain[offset_mask]
            offset_stress = offset_stress[offset_mask]

    proof_strain, proof_stress = _proof_point(
        strain,
        stress,
        offset_strain,
        offset_stress,
        min_strain=_proof_search_start_strain(strain, start_idx, end_idx),
    )

    display_start_idx = _tension_display_start_idx(strain, stress)
    if display_start_idx > 0:
        strain = strain[display_start_idx:]
        stress = stress[display_start_idx:]
        if start_idx is not None:
            start_idx = max(int(start_idx) - display_start_idx, 0)
        if end_idx is not None:
            end_idx = max(int(end_idx) - display_start_idx, -1)

    line_start_idx = _line_display_start_idx(strain, stress, start_idx, end_idx, slope, intercept)
    offset_min_display_strain = float(strain[line_start_idx]) if strain.size > 0 else None

    stress_mpa = stress / 1e6
    offset_stress_mpa = offset_stress / 1e6
    yield_stress_mpa = yield_stress / 1e6 if np.isfinite(yield_stress) else np.nan
    proof_stress_mpa = proof_stress / 1e6 if np.isfinite(proof_stress) else np.nan
    fracture_stress_mpa = fracture_stress / 1e6 if np.isfinite(fracture_stress) else np.nan
    slope_mpa = slope / 1e6 if np.isfinite(slope) else np.nan
    intercept_mpa = intercept / 1e6 if np.isfinite(intercept) else np.nan
    offset_marker_strain = proof_strain if np.isfinite(proof_strain) else yield_strain
    offset_marker_stress_mpa = (
        proof_stress_mpa if np.isfinite(proof_stress_mpa) else yield_stress_mpa
    )

    offset_strain, offset_stress_mpa = _format_offset_curve_for_marker_display(
        offset_strain,
        offset_stress_mpa,
        offset_marker_strain,
        offset_marker_stress_mpa,
        min_display_stress=float(np.nanmin(stress_mpa)) if stress_mpa.size > 0 else None,
        min_display_strain=offset_min_display_strain,
    )

    has_linear_region = (
        start_idx is not None
        and end_idx is not None
        and 0 <= start_idx <= end_idx < len(strain)
        and np.isfinite(slope_mpa)
        and np.isfinite(intercept_mpa)
    )

    return {
        "strain": strain,
        "stress": stress_mpa,
        "start_idx": start_idx,
        "end_idx": end_idx,
        "line_start_idx": line_start_idx,
        "slope": slope_mpa,
        "intercept": intercept_mpa,
        "offset_strain": offset_strain,
        "offset_stress": offset_stress_mpa,
        "yield_strain": float(yield_strain) if np.isfinite(yield_strain) else np.nan,
        "yield_stress": float(yield_stress_mpa) if np.isfinite(yield_stress_mpa) else np.nan,
        "proof_strain": float(proof_strain) if np.isfinite(proof_strain) else np.nan,
        "proof_stress": float(proof_stress_mpa) if np.isfinite(proof_stress_mpa) else np.nan,
        "fracture_strain": float(fracture_strain) if np.isfinite(fracture_strain) else np.nan,
        "fracture_stress": float(fracture_stress_mpa) if np.isfinite(fracture_stress_mpa) else np.nan,
        "has_linear_region": has_linear_region,
    }


def _marker_style_for_petg(label: str):
    if label == "First Max Point":
        return dict(size=11, color="darkorange", symbol="diamond")
    if label == "0.2% Offset Intersection":
        return dict(size=11, color="red", symbol="circle-open")
    return dict(size=11, color="red")


def _build_variant_figure(variant, *, sample_title: str, yield_label: str, proof_label: str, prepared_data):
    import plotly.graph_objects as go

    strain = prepared_data["strain"]
    stress = prepared_data["stress"]
    x_series = []
    y_series = []

    fig = go.Figure()

    if "stress_strain_curve" in variant.component_keys:
        fig.add_trace(go.Scatter(x=strain, y=stress, mode="lines", name="Stress-Strain Curve"))
        x_series.append(strain)
        y_series.append(stress)

    if prepared_data["has_linear_region"] and "linear_region" in variant.component_keys:
        line_start_idx = prepared_data["line_start_idx"]
        end_idx = int(prepared_data["end_idx"])
        x_linear_region = strain[line_start_idx : end_idx + 1]
        y_linear_region = stress[line_start_idx : end_idx + 1]
        fig.add_trace(
            go.Scatter(
                x=x_linear_region,
                y=y_linear_region,
                mode="lines",
                line=dict(width=6),
                name="Linear Region",
            )
        )
        x_series.append(x_linear_region)
        y_series.append(y_linear_region)

    if prepared_data["has_linear_region"] and "linear_fit" in variant.component_keys:
        line_start_idx = prepared_data["line_start_idx"]
        end_idx = int(prepared_data["end_idx"])
        x_linear_fit = strain[line_start_idx : end_idx + 1]
        y_linear_fit = prepared_data["slope"] * x_linear_fit + prepared_data["intercept"]
        fig.add_trace(
            go.Scatter(
                x=x_linear_fit,
                y=y_linear_fit,
                mode="lines",
                line=dict(dash="dash"),
                name="Linear Fit",
            )
        )
        x_series.append(x_linear_fit)
        y_series.append(y_linear_fit)

    if (
        "offset_curve" in variant.component_keys
        and prepared_data["offset_strain"].size > 0
        and prepared_data["offset_stress"].size > 0
    ):
        fig.add_trace(
            go.Scatter(
                x=prepared_data["offset_strain"],
                y=prepared_data["offset_stress"],
                mode="lines",
                name="Offset Curve",
            )
        )
        x_series.append(prepared_data["offset_strain"])
        y_series.append(prepared_data["offset_stress"])

    show_proof_marker = (
        "yield_point" in variant.component_keys
        and np.isfinite(prepared_data["proof_strain"])
        and np.isfinite(prepared_data["proof_stress"])
    )
    if show_proof_marker:
        proof_x = np.array([prepared_data["proof_strain"]], dtype=float)
        proof_y = np.array([prepared_data["proof_stress"]], dtype=float)
        fig.add_trace(
            go.Scatter(
                x=proof_x,
                y=proof_y,
                mode="markers",
                marker=_marker_style_for_petg(proof_label),
                name=proof_label,
            )
        )
        x_series.append(proof_x)
        y_series.append(proof_y)

    if (
        "yield_point" in variant.component_keys
        and np.isfinite(prepared_data["yield_strain"])
        and np.isfinite(prepared_data["yield_stress"])
        and (yield_label != proof_label or not show_proof_marker)
    ):
        yield_x = np.array([prepared_data["yield_strain"]], dtype=float)
        yield_y = np.array([prepared_data["yield_stress"]], dtype=float)
        fig.add_trace(
            go.Scatter(
                x=yield_x,
                y=yield_y,
                mode="markers",
                marker=_marker_style_for_petg(yield_label),
                name=yield_label,
            )
        )
        x_series.append(yield_x)
        y_series.append(yield_y)

    if (
        "fracture_point" in variant.component_keys
        and np.isfinite(prepared_data["fracture_strain"])
        and np.isfinite(prepared_data["fracture_stress"])
    ):
        fracture_x = np.array([prepared_data["fracture_strain"]], dtype=float)
        fracture_y = np.array([prepared_data["fracture_stress"]], dtype=float)
        fig.add_trace(
            go.Scatter(
                x=fracture_x,
                y=fracture_y,
                mode="markers",
                marker=dict(size=11, color="black", symbol="x"),
                name="Fracture Point",
            )
        )
        x_series.append(fracture_x)
        y_series.append(fracture_y)

    x_min, x_max = _finite_plot_bounds(*x_series, fallback=(0.0, 1.0))
    y_min, y_max = _finite_plot_bounds(*y_series, fallback=(0.0, 1.0))
    x_pad = 0.05 * max(x_max - x_min, 1.0e-6)
    y_pad = 0.05 * max(y_max - y_min, 1.0e-6)

    fig.update_layout(
        title=f"PETG Tension - {sample_title} - {variant.title}",
        title_x=0.5,
        xaxis_title="Strain",
        yaxis_title="Stress (MPa)",
        template="plotly_white",
        xaxis=dict(range=[x_min - x_pad, x_max + x_pad]),
        yaxis=dict(range=[y_min - y_pad, y_max + y_pad]),
    )
    return fig


def _write_petg_html(figure, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(path, include_plotlyjs="directory")


def render_petg_tension_graphs(
    *,
    sample_name: str,
    display_name: str,
    graph_path: Path,
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
    yield_method: str,
    write_component_graphs: bool = True,
):
    fracture_strain, fracture_stress = _fracture_point(strain, stress)
    prepared_data = _prepare_petg_plot_data(
        strain=strain,
        stress=stress,
        start_idx=start_idx,
        end_idx=end_idx,
        slope=slope,
        intercept=intercept,
        offset_strain=offset_strain,
        offset_stress=offset_stress,
        yield_strain=yield_strain,
        yield_stress=yield_stress,
        fracture_strain=fracture_strain,
        fracture_stress=fracture_stress,
    )
    yield_label = _yield_label_for_petg(yield_method)
    proof_label = _proof_marker_label_for_petg(yield_method)

    graph_path.parent.mkdir(parents=True, exist_ok=True)
    overview_variant = next(
        variant
        for variant in PETG_TENSION_GRAPH_VARIANTS
        if variant.key == "full_overview"
    )
    overview_figure = _build_variant_figure(
        overview_variant,
        sample_title=display_name,
        yield_label=yield_label,
        proof_label=proof_label,
        prepared_data=prepared_data,
    )
    _write_petg_html(overview_figure, graph_path)

    component_graph_paths = {}
    manifest_path = None
    if write_component_graphs:
        component_root = graph_path.parent / "COMPONENTS" / canonical_sample_name(sample_name)
        component_root.mkdir(parents=True, exist_ok=True)

        overview_component_path = component_root / overview_variant.file_name
        shutil.copyfile(graph_path, overview_component_path)
        component_graph_paths[overview_variant.key] = overview_component_path

        for variant in PETG_TENSION_GRAPH_VARIANTS:
            if variant.key == overview_variant.key:
                continue
            save_path = component_root / variant.file_name
            figure = _build_variant_figure(
                variant,
                sample_title=display_name,
                yield_label=yield_label,
                proof_label=proof_label,
                prepared_data=prepared_data,
            )
            _write_petg_html(figure, save_path)
            component_graph_paths[variant.key] = save_path

        manifest_path = component_root / "graph_manifest.json"
        manifest_payload = {
            "sample_name": canonical_sample_name(sample_name),
            "display_name": display_name,
            "offset_strain": PETG_TENSION_OFFSET_STRAIN,
            "yield_label": yield_label,
            "graphs": describe_petg_tension_graph_system(),
        }
        manifest_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")

    return PetgTensionRenderedGraphs(
        primary_graph_path=graph_path,
        component_graph_paths=component_graph_paths,
        manifest_path=manifest_path,
    )
