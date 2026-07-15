"""Deterministic domain primitives shared by every adapter and workflow."""

from list_engine.core.models import Company
from list_engine.core.piva import InvalidPIVA, is_valid_piva, normalize_piva
from list_engine.core.repository import (
    InMemoryCompanyRepository,
    UpsertAction,
    UpsertResult,
    deduplicate_companies,
    merge_companies,
)

__all__ = [
    "Company",
    "InMemoryCompanyRepository",
    "InvalidPIVA",
    "UpsertAction",
    "UpsertResult",
    "deduplicate_companies",
    "is_valid_piva",
    "merge_companies",
    "normalize_piva",
]
