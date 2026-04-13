# Project Structure

The repo root is intentionally thin. Most implementation lives inside folders by responsibility.

## Analysis Packages

- `compression_core/`
  - compression I/O, candidate selection, subgroup correction, yielding, results, and pipeline
- `tension_core/`
  - tension I/O, results, and pipeline
- `petg_tension_core/`
  - PETG-only tension graph components and diagnostic graph generation
- `bending_core/`
  - 4-point bending I/O, results, and pipeline
- `shear_core/`
  - shear I/O, pipeline internals, results, and the dedicated shear system boundary

## Shared Packages

- `shared_core/`
  - shared elastic-fit logic, curve helpers, and linear-region search
- `reporting_core/`
  - workbook sorting, summary generation, and highlighting
- `app_core/`
  - top-level orchestration and the project runner
- `verification/`
  - triage helpers for test/result vetting

## Assets And Data

- `resources/`
  - workbook assets used by the analysis loaders
- `DATA/`
  - raw source test data
- `GRAPHS/`
  - generated HTML plots
- `STATS/`
  - generated workbook outputs
