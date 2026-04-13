from __future__ import annotations

from dataclasses import dataclass


PETG_TENSION_OFFSET_STRAIN = 0.002


@dataclass(frozen=True)
class PetgTensionGraphComponent:
    key: str
    label: str
    description: str


@dataclass(frozen=True)
class PetgTensionGraphVariant:
    key: str
    file_name: str
    title: str
    description: str
    component_keys: tuple[str, ...]


PETG_TENSION_GRAPH_COMPONENTS = (
    PetgTensionGraphComponent(
        key="stress_strain_curve",
        label="Stress-Strain Curve",
        description="The stitched PETG tension curve used as the base trace for every review plot.",
    ),
    PetgTensionGraphComponent(
        key="linear_region",
        label="Linear Region",
        description="The elastic window selected for the modulus fit.",
    ),
    PetgTensionGraphComponent(
        key="linear_fit",
        label="Linear Fit",
        description="The best-fit elastic line across the selected linear region.",
    ),
    PetgTensionGraphComponent(
        key="offset_curve",
        label="0.2% Offset Curve",
        description="The proof-stress reference line generated with an offset of 0.002 strain.",
    ),
    PetgTensionGraphComponent(
        key="yield_point",
        label="Yield Point",
        description="The PETG proof-intersection marker, plus the first-maximum marker when peak fallback review is needed.",
    ),
    PetgTensionGraphComponent(
        key="fracture_point",
        label="Fracture Point",
        description="The last point before the tension curve is treated as failed or fractured.",
    ),
)


PETG_TENSION_GRAPH_VARIANTS = (
    PetgTensionGraphVariant(
        key="full_overview",
        file_name="00_full_overview.html",
        title="Full Overview",
        description="Combined PETG diagnostic view with the stress-strain curve, elastic fit, offset curve, proof-intersection marker, optional first-maximum marker, and fracture marker.",
        component_keys=(
            "stress_strain_curve",
            "linear_region",
            "linear_fit",
            "offset_curve",
            "yield_point",
            "fracture_point",
        ),
    ),
    PetgTensionGraphVariant(
        key="stress_strain_only",
        file_name="01_stress_strain_curve.html",
        title="Stress-Strain Curve",
        description="Raw PETG stress-strain trace without overlays.",
        component_keys=("stress_strain_curve",),
    ),
    PetgTensionGraphVariant(
        key="elastic_fit_review",
        file_name="02_linear_region_and_fit.html",
        title="Elastic Fit Review",
        description="Stress-strain trace with the chosen linear region and linear fit only.",
        component_keys=("stress_strain_curve", "linear_region", "linear_fit"),
    ),
    PetgTensionGraphVariant(
        key="offset_yield_review",
        file_name="03_offset_yield_review.html",
        title="Offset Yield Review",
        description="Stress-strain trace with the linear fit, 0.2% offset curve, the proof-intersection marker, and the PETG first-maximum marker when applicable.",
        component_keys=(
            "stress_strain_curve",
            "linear_region",
            "linear_fit",
            "offset_curve",
            "yield_point",
        ),
    ),
    PetgTensionGraphVariant(
        key="fracture_review",
        file_name="04_fracture_review.html",
        title="Fracture Review",
        description="Stress-strain trace focused on where the PETG analysis ends at fracture.",
        component_keys=("stress_strain_curve", "fracture_point"),
    ),
)


_COMPONENT_LABELS = {
    component.key: component.label
    for component in PETG_TENSION_GRAPH_COMPONENTS
}


def describe_petg_tension_graph_system():
    descriptions = []
    for variant in PETG_TENSION_GRAPH_VARIANTS:
        descriptions.append(
            {
                "key": variant.key,
                "file_name": variant.file_name,
                "title": variant.title,
                "description": variant.description,
                "components": [
                    _COMPONENT_LABELS[key]
                    for key in variant.component_keys
                ],
            }
        )
    return descriptions

