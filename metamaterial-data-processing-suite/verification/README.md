# Test Vetting Loop

Use this loop for each test file:

1. Run one test file.
2. If it passes, mark it as `passed` and move on.
3. If it fails, classify the failure before changing code.
4. Apply the fix strategy for that category.
5. Rerun the same test file and classify again until it passes or the category changes.

## Categories

- `A_environment_or_dependency`
  Fix interpreter, package, or dependency issues first.
- `B_import_or_api_break`
  Fix exports, imports, file moves, renamed symbols, or function signatures.
- `C_behavior_or_assertion`
  Fix the actual product logic or test expectations.
- `D_data_fixture_or_io`
  Fix fixture setup, workbook/data assumptions, file paths, or I/O behavior.
- `E_unknown`
  Manually inspect and assign the nearest category before making more changes.

## Rule

Do not jump straight into product-logic edits if the failure is really category A or B. Reclassify after every rerun.
