from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from reporting_core import finalize_results_workbook
from shared_core.adaptive_fit import ElasticSampleAnalysis, format_sample_display_name

from .decisions import resolve_shear_subgroup_fit_decisions
from .io import DEFAULT_SHEAR_DIMENSIONS_WORKBOOK
from .pipeline import _is_expected_sample_skip, analyze_shear_file, finalize_shear_metrics
from .results import get_subgroup, save_results_to_xlsx, write_shear_results


@dataclass(frozen=True)
class ShearSystemPaths:
    project_root: Path
    stats_root: Path
    graphs_root: Path
    dimensions_workbook: Path

    @classmethod
    def from_project_root(cls, project_root: Path | None = None) -> "ShearSystemPaths":
        root = Path(project_root) if project_root is not None else Path(__file__).resolve().parent.parent
        return cls(
            project_root=root,
            stats_root=root / "STATS",
            graphs_root=root / "GRAPHS",
            dimensions_workbook=root / "resources" / DEFAULT_SHEAR_DIMENSIONS_WORKBOOK,
        )


def build_default_shear_paths(project_root: Path | None = None) -> ShearSystemPaths:
    return ShearSystemPaths.from_project_root(project_root)


class ShearAnalysisSystem:
    def __init__(self, paths: ShearSystemPaths | None = None):
        self.paths = paths or build_default_shear_paths()

    def analyze_file(self, file_name: Path) -> ElasticSampleAnalysis:
        return analyze_shear_file(
            file_name,
            dimensions_workbook=self.paths.dimensions_workbook,
        )

    def results_workbook_path(self, material: str) -> Path:
        return self.paths.stats_root / material / f"{material} SHEAR RESULTS.xlsx"

    def graph_path_for_sample(self, sample: ElasticSampleAnalysis) -> Path:
        graph_name = format_sample_display_name(sample.sample_name, sample.is_valid, sample.decision_reason)
        return self.paths.graphs_root / sample.material / "SHEAR" / f"{graph_name}.html"

    def finalize_results_workbook(self, material: str) -> None:
        finalize_results_workbook(self.results_workbook_path(material))

    def process_file(self, file_name: Path) -> ElasticSampleAnalysis:
        sample = self.analyze_file(file_name)
        resolve_shear_subgroup_fit_decisions([sample])
        row = finalize_shear_metrics(sample, self.paths.graphs_root)
        plot_path = self.graph_path_for_sample(sample)
        save_results_to_xlsx(
            sample_name=row[0],
            E=np.nan if row[1] is None or not np.isfinite(row[1]) else float(row[1]) * 1e9,
            yield_strength=np.nan if row[2] is None or not np.isfinite(row[2]) else float(row[2]) * 1e6,
            ultimate_strength=np.nan if row[3] is None or not np.isfinite(row[3]) else float(row[3]) * 1e6,
            output_path=self.results_workbook_path(sample.material),
            plot_path=plot_path,
            is_valid=sample.is_valid,
            fit_start_strain=row[6],
            fit_end_strain=row[7],
            fit_mode=row[8],
            decision_reason=row[9],
        )
        self.finalize_results_workbook(sample.material)
        return sample

    def process_folder(self, folder: Path) -> list[ElasticSampleAnalysis]:
        grouped_samples: dict[str, list[ElasticSampleAnalysis]] = {}
        if not folder.is_dir():
            print(f"Skipped shear batch: folder not found -> {folder}")
            return []

        files = sorted(folder.glob("*.xlsx"))
        if not files:
            print(f"Skipped shear batch: no .xlsx files found in -> {folder}")
            return []

        for file in files:
            try:
                sample = self.analyze_file(file)
            except Exception as error:
                label = "Skipped" if _is_expected_sample_skip(error) else "Failed"
                print(f"{label}: {file.name} -> {error}")
                continue
            subgroup = get_subgroup(sample.sample_name)
            grouped_samples.setdefault(subgroup, []).append(sample)

        all_samples = []
        for subgroup in sorted(grouped_samples):
            subgroup_samples = grouped_samples[subgroup]
            resolve_shear_subgroup_fit_decisions(subgroup_samples)
            all_samples.extend(subgroup_samples)

        if not all_samples:
            return []

        rows = [finalize_shear_metrics(sample, self.paths.graphs_root) for sample in all_samples]
        material = all_samples[0].material
        write_shear_results(self.results_workbook_path(material), rows)
        self.finalize_results_workbook(material)
        return all_samples


def build_default_shear_system(project_root: Path | None = None) -> ShearAnalysisSystem:
    return ShearAnalysisSystem(paths=build_default_shear_paths(project_root))
