# Rejected Sample Triage

Current state after the latest compression and shared-fit recovery changes.

## Old rejected graphs that are now recoverable

- `N0TN100 - 2`
  Category: `short_window_fallback`
- `N0TN100C - 2`
  Category: `alternate_candidate`
- `P0CP70W - 2`
  Category: `short_ramp_relaxed`
- `P1CP40W - 2`
  Category: `short_ramp_exception`
- `P1CP70W - 2`
  Category: `short_ramp_relaxed`
- `P1CP70W - 3`
  Category: `short_ramp_exception`
- `P1CP70S - 2`
  Category: `manual_exception`
- `P1CP70W - 1`
  Category: `manual_exception`
- `P1CP60S - 3`
  Category: `manual_exception`
- `P1TN40 - 4`
  Category: `alternate_candidate`

These graph files are stale. A fresh batch run should regenerate them as accepted samples.

## Current remaining rejects

- `N1CP100S - 8`
  Category: `no_stable_pre_yield_window`
  Reason: no usable compression candidate survived, even under looser short-window probing.
- `P0TN40 - 2`
  Category: `late_tail_false_fit`
  Reason: candidate windows start far later than sibling elastic windows and never approach sibling modulus values.
- `P1TN100 - 3`
  Category: `tension_high_band_outlier`
  Reason: all candidates stay high relative to the subgroup, around `0.897-1.208 GPa` vs siblings around `0.619-0.693 GPa`.

## Current live reject count

- Compression: `1`
- Tension: `2`
- Total: `3`
