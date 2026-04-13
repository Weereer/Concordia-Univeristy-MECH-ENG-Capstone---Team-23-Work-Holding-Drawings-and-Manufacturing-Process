# Shear System

The shear workflow is now split into a dedicated application boundary inside `shear_core/system.py`.

## Why

- Shear processing has its own dimensions workbook, graph output, and stats workbook behavior.
- Keeping those dependencies inside the shear package reduces the chance that app-level changes break shear runs.
- The runner should invoke shear as a batch system, not reimplement its file loop.

## Structure

- `shear_core/io.py`
  - reads raw shear workbooks and dimensions data
  - accepts the dimensions workbook path as an explicit dependency
- `shear_core/pipeline.py`
  - contains shear analysis internals such as fit selection and graph/result row generation
- `shear_core/results.py`
  - builds and writes shear result rows and workbooks
- `shear_core/system.py`
  - owns the configurable paths for dimensions, graphs, and stats
  - exposes `ShearAnalysisSystem.process_file()` and `ShearAnalysisSystem.process_folder()`
  - finalizes the shear results workbook immediately after writing it so the summary table is always present
- `app_core/runner.py`
  - delegates shear batch execution to `shear_core` instead of managing individual files itself

## Default Paths

`ShearSystemPaths.from_project_root()` resolves:

- dimensions workbook: `resources/PET G Sample Dimensions.xlsx`
- graph root: `GRAPHS/`
- stats root: `STATS/`

Tests can inject alternate paths by constructing `ShearSystemPaths` directly.

## Shear Acceptance Rules

Shear uses its own fit-selection and subgroup-decision tuning instead of the stricter shared defaults:

- shorter minimum fit windows
- a lower stress gate for fit search
- a lower minimum `R^2` for acceptable shear fits
- a wider fallback subgroup band for small shear groups

Those settings are defined in `shear_core/constants.py` and applied through `shear_core/decisions.py`.
