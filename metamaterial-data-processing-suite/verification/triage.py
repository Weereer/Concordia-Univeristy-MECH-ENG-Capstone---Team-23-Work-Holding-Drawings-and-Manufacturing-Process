from __future__ import annotations

from dataclasses import dataclass


STATUS_PASSED = "passed"
STATUS_FAILED = "failed"

CATEGORY_A_ENVIRONMENT = "A_environment_or_dependency"
CATEGORY_B_INTERFACE = "B_import_or_api_break"
CATEGORY_C_LOGIC = "C_behavior_or_assertion"
CATEGORY_D_DATA = "D_data_fixture_or_io"
CATEGORY_E_UNKNOWN = "E_unknown"


@dataclass(frozen=True)
class TestTriage:
    status: str
    category: str
    reason: str
    fix_strategy: str


def _strategy_for(category: str) -> str:
    if category == CATEGORY_A_ENVIRONMENT:
        return "Fix the environment first: repair packages, imports, interpreter, or missing dependencies before touching product logic."
    if category == CATEGORY_B_INTERFACE:
        return "Fix module layout, exports, renamed symbols, or call signatures, then rerun the same test file."
    if category == CATEGORY_C_LOGIC:
        return "Fix behavior or assertions in the code path under test, then rerun and re-evaluate the category."
    if category == CATEGORY_D_DATA:
        return "Fix fixtures, workbook/data assumptions, path handling, or I/O setup, then rerun the same test file."
    return "Inspect the raw failure, assign the closest category, fix it, and rerun."


def categorize_test_outcome(output: str, returncode: int) -> TestTriage:
    if returncode == 0:
        return TestTriage(
            status=STATUS_PASSED,
            category=STATUS_PASSED,
            reason="The test file completed without failures.",
            fix_strategy="No code change needed for this test file.",
        )

    text = (output or "").lower()

    environment_markers = (
        "modulenotfounderror",
        "no module named",
        "dll load failed",
        "namespace package",
        "site-packages",
        "pip",
        "cannot import name 'load_workbook' from 'openpyxl'",
    )
    interface_markers = (
        "cannot import name",
        "attributeerror",
        "typeerror",
        "unexpected keyword argument",
        "missing 1 required positional argument",
    )
    logic_markers = (
        "assertionerror",
        "failed (failures=",
        "failed (errors=",
        "not equal to tolerance",
        "assert",
    )
    data_markers = (
        "filenotfounderror",
        "permissionerror",
        "valueerror",
        "sample '",
        "read_excel",
        "workbook",
        "sheet",
        "path",
    )

    if any(marker in text for marker in environment_markers):
        category = CATEGORY_A_ENVIRONMENT
        reason = "The failure looks like an interpreter or dependency problem rather than a product-logic regression."
    elif any(marker in text for marker in interface_markers):
        category = CATEGORY_B_INTERFACE
        reason = "The failure looks like an import, symbol, or API contract break introduced by code organization changes."
    elif any(marker in text for marker in logic_markers):
        category = CATEGORY_C_LOGIC
        reason = "The test ran far enough to assert behavior, so the failure is most likely in the implementation logic."
    elif any(marker in text for marker in data_markers):
        category = CATEGORY_D_DATA
        reason = "The failure looks tied to fixtures, workbook/data assumptions, or file I/O."
    else:
        category = CATEGORY_E_UNKNOWN
        reason = "The failure did not match a known pattern and still needs manual triage."

    return TestTriage(
        status=STATUS_FAILED,
        category=category,
        reason=reason,
        fix_strategy=_strategy_for(category),
    )
