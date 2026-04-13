from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class CompressionFitCandidate:
    slope: float
    intercept: float
    r2: float
    start_idx: int
    end_idx: int
    start_strain: float
    end_strain: float
    slope_cv: float
    zero_strain: float
    fit_mode: str

    @property
    def window_points(self) -> int:
        return self.end_idx - self.start_idx + 1


@dataclass
class CompressionSampleAnalysis:
    sample_name: str
    file_path: Path
    material: str
    strain: np.ndarray
    stress: np.ndarray
    candidates: list[CompressionFitCandidate] = field(default_factory=list)
    yield_cutoff_idx: int | None = None
    geometric_instability: bool = False
    selected_candidate: CompressionFitCandidate | None = None
    is_valid: bool = False
    fit_mode: str = ""
    decision_reason: str = ""
