from __future__ import annotations

import pytest

from list_engine.core.piva import (
    InvalidPIVA,
    calculate_check_digit,
    is_valid_piva,
    normalize_piva,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("99000000002", "99000000002"),
        ("IT 990.000.000-02", "99000000002"),
        ("it/99000000002", "99000000002"),
    ],
)
def test_normalize_piva_accepts_controlled_source_formatting(raw: str, expected: str) -> None:
    assert normalize_piva(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "99000000003",
        "GB99000000002",
        "9900000000",
        "990000000020",
        "VAT99000000002",
        "\uff19\uff19\uff10\uff10\uff10\uff10\uff10\uff10\uff10\uff10\uff12",
        "\u0669\u0669\u0660\u0660\u0660\u0660\u0660\u0660\u0660\u0660\u0662",
    ],
)
def test_normalize_piva_rejects_uncertain_identity(raw: str) -> None:
    with pytest.raises(InvalidPIVA):
        normalize_piva(raw)


def test_normalize_piva_rejects_numeric_input_to_preserve_leading_zeroes() -> None:
    with pytest.raises(InvalidPIVA):
        normalize_piva(99000000002)  # type: ignore[arg-type]


def test_checksum_calculation_is_exposed_for_seed_and_import_diagnostics() -> None:
    assert calculate_check_digit("9900000000") == 2
    assert is_valid_piva("99000000002")
    assert not is_valid_piva(99000000002)


@pytest.mark.parametrize(
    "stem",
    [
        "\uff19\uff19\uff10\uff10\uff10\uff10\uff10\uff10\uff10\uff10",
        "\u0669\u0669\u0660\u0660\u0660\u0660\u0660\u0660\u0660\u0660",
    ],
)
def test_checksum_rejects_unicode_numerals_for_database_parity(stem: str) -> None:
    with pytest.raises(InvalidPIVA):
        calculate_check_digit(stem)
