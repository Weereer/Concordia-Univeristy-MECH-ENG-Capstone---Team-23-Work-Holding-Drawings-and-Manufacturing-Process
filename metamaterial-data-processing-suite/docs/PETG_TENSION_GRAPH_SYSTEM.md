# PETG Tension Graph System

`petg_tension_core/` is an independent PETG-only graphing package. It exists so PETG tension behavior can be debugged without changing Nylon tension plots at the same time.

## Offset

- The PETG proof offset is `0.002` strain, which is a 0.2% offset.
- `0.02` would be a 2% offset and is 10x larger than the current PETG tension proof line.

## Graph Components

- `Stress-Strain Curve`
  The stitched PETG tension trace.
- `Linear Region`
  The elastic window chosen for the modulus fit.
- `Linear Fit`
  The best-fit line across the selected elastic window.
- `0.2% Offset Curve`
  The proof-stress reference line shifted by `0.002` strain.
- `Yield Point`
  The proof-intersection marker used to review the `0.2%` offset method.
- `First Max Point`
  When PETG falls back to a peak-based review, the first maximum is also kept on the overlay so the proof intersection and peak can be compared directly.
- `Fracture Point`
  The last point before the PETG curve is treated as failed/fractured.

## Generated PETG Graph Files

- `00_full_overview.html`
  Stress-strain + linear region + linear fit + offset curve + proof intersection + optional first max point + fracture point.
- `01_stress_strain_curve.html`
  Stress-strain curve only.
- `02_linear_region_and_fit.html`
  Stress-strain + linear region + linear fit.
- `03_offset_yield_review.html`
  Stress-strain + linear region + linear fit + offset curve + proof intersection + optional first max point.
- `04_fracture_review.html`
  Stress-strain + fracture point.

These component graphs are written under `GRAPHS/PETG/TENSION/COMPONENTS/<sample name>/` and the workbook still links to the main PETG tension HTML graph.

## Command

- Default PETG tension run:
  `.\.codex_env\Scripts\python.exe -m app_core.petg_tension_graphs`
- Custom PETG tension folder:
  `.\.codex_env\Scripts\python.exe -m app_core.petg_tension_graphs "C:\path\to\PETG TENSION"`
- Faster overview-only run:
  `.\.codex_env\Scripts\python.exe -m app_core.petg_tension_graphs --overview-only`
