"""Italian VAT-number normalisation and deterministic checksum validation."""

from __future__ import annotations

import re

_SEPARATORS = re.compile(r"[\s.\-_/]+")
_ASCII_TEN_DIGITS = re.compile(r"[0-9]{10}")
_ASCII_ELEVEN_DIGITS = re.compile(r"[0-9]{11}")


class InvalidPIVA(ValueError):
    """Raised when a value cannot be normalised to a valid Italian P.IVA."""


def calculate_check_digit(first_ten_digits: str) -> int:
    """Return the Italian VAT check digit for a ten-digit stem.

    The algorithm is deterministic and defined on digits only. It is deliberately
    separate from :func:`normalize_piva` so seeds and import diagnostics can test the
    checksum without accepting malformed source values.
    """

    if _ASCII_TEN_DIGITS.fullmatch(first_ten_digits) is None:
        raise InvalidPIVA("P.IVA checksum input must contain exactly 10 digits")

    total = sum(int(first_ten_digits[index]) for index in range(0, 10, 2))
    for index in range(1, 10, 2):
        doubled = int(first_ten_digits[index]) * 2
        total += doubled - 9 if doubled > 9 else doubled
    return (10 - total % 10) % 10


def normalize_piva(value: str) -> str:
    """Normalise and validate an Italian VAT number.

    Accepted source formatting includes an optional ``IT`` prefix and common visual
    separators. Arbitrary letters are rejected rather than silently stripped because
    identity resolution is a certainty boundary in List Engine.
    """

    if not isinstance(value, str):
        raise InvalidPIVA("P.IVA must be supplied as text so leading zeroes are preserved")

    compact = _SEPARATORS.sub("", value.strip()).upper()
    if compact.startswith("IT"):
        compact = compact[2:]
    if _ASCII_ELEVEN_DIGITS.fullmatch(compact) is None:
        raise InvalidPIVA("P.IVA must contain exactly 11 digits after normalisation")

    expected = calculate_check_digit(compact[:10])
    if int(compact[-1]) != expected:
        raise InvalidPIVA("P.IVA checksum is invalid")
    return compact


def is_valid_piva(value: object) -> bool:
    """Return ``True`` only when ``value`` is a checksum-valid Italian P.IVA string."""

    if not isinstance(value, str):
        return False
    try:
        normalize_piva(value)
    except InvalidPIVA:
        return False
    return True
