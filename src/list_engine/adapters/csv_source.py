"""Strict deterministic CSV adapter for cold company ingestion."""

from __future__ import annotations

import csv
import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from io import StringIO
from types import MappingProxyType

from pydantic import ValidationError

from list_engine.core.models import Company
from list_engine.core.piva import InvalidPIVA, normalize_piva
from list_engine.ingestion import (
    IssueSeverity,
    MappedCompanyRecord,
    MappedSourceBatch,
    QualityIssue,
    QualityReport,
    RawRecordEnvelope,
    SchemaContract,
    SourceReadStatus,
)
from list_engine.ingestion.contracts import SourceSchemaObservation

CanonicalValue = str | Decimal | int | datetime | None
_ITALIAN_DECIMAL = re.compile(r"[+-]?(?:[0-9]{1,3}(?:\.[0-9]{3})+|[0-9]+)(?:,[0-9]+)?")
_TRUSTED_FIELDS = frozenset({"source", "is_demo"})


class CSVSourceError(ValueError):
    """Base class for CSV boundary failures."""


class CSVHeaderDriftError(CSVSourceError):
    """The source header no longer matches its ingestion contract."""

    def __init__(
        self,
        *,
        missing: tuple[str, ...] = (),
        unexpected: tuple[str, ...] = (),
        duplicated: tuple[str, ...] = (),
        observation: SourceSchemaObservation,
    ) -> None:
        self.missing = missing
        self.unexpected = unexpected
        self.duplicated = duplicated
        self.observation = observation
        parts = []
        if missing:
            parts.append(f"missing={','.join(missing)}")
        if unexpected:
            parts.append(f"unexpected={','.join(unexpected)}")
        if duplicated:
            parts.append(f"duplicated={','.join(duplicated)}")
        super().__init__("CSV header drift: " + "; ".join(parts or ["header row is missing"]))


class CSVFormatError(CSVSourceError):
    """The CSV structure cannot be parsed safely."""


class CSVFieldError(CSVSourceError):
    """A row field cannot be normalised without guessing."""

    def __init__(self, field_name: str, message: str) -> None:
        self.field_name = field_name
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class CSVContract:
    """Exact source-header to canonical-field ingestion contract."""

    source: str
    header_mapping: Mapping[str, str]
    required_fields: frozenset[str] = field(
        default_factory=lambda: frozenset({"piva", "legal_name", "source_observed_at"})
    )
    decimal_fields: frozenset[str] = field(default_factory=lambda: frozenset({"revenue_eur"}))
    integer_fields: frozenset[str] = field(default_factory=lambda: frozenset({"employees"}))
    datetime_fields: frozenset[str] = field(
        default_factory=lambda: frozenset({"source_observed_at", "observed_at"})
    )
    delimiter: str = ";"
    schema_contract: SchemaContract = field(init=False, repr=False)

    def __post_init__(self) -> None:
        source = self.source.strip()
        mapping = dict(self.header_mapping)
        if not source:
            raise ValueError("CSV source name cannot be blank")
        if len(self.delimiter) != 1:
            raise ValueError("CSV delimiter must be exactly one character")
        if not mapping or any(not key or not value for key, value in mapping.items()):
            raise ValueError("CSV header mapping cannot contain blank names")
        targets = tuple(mapping.values())
        duplicates = sorted(name for name, count in Counter(targets).items() if count > 1)
        if duplicates:
            raise ValueError(f"Canonical fields mapped more than once: {','.join(duplicates)}")
        reserved = _TRUSTED_FIELDS.intersection(targets)
        if reserved:
            raise ValueError(f"CSV cannot control trusted fields: {','.join(sorted(reserved))}")
        missing = self.required_fields.difference(targets)
        if missing:
            missing_names = ",".join(sorted(missing))
            raise ValueError(f"CSV contract does not map required fields: {missing_names}")
        seen: set[str] = set()
        for group in (self.decimal_fields, self.integer_fields, self.datetime_fields):
            overlap = seen.intersection(group)
            if overlap:
                raise ValueError(f"CSV fields have conflicting types: {','.join(sorted(overlap))}")
            seen.update(group)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "header_mapping", MappingProxyType(mapping))
        object.__setattr__(
            self,
            "schema_contract",
            SchemaContract(source=source, required_fields=frozenset(mapping)),
        )


@dataclass(frozen=True, slots=True)
class CSVRawRow:
    """Exact cells retained for provenance and quarantine."""

    row_number: int
    headers: tuple[str, ...]
    values: tuple[str, ...]

    def as_mapping(self) -> Mapping[str, str]:
        return MappingProxyType(dict(zip(self.headers, self.values, strict=False)))


@dataclass(frozen=True, slots=True)
class CSVParsedRow:
    raw: CSVRawRow
    values: Mapping[str, CanonicalValue]


@dataclass(frozen=True, slots=True)
class CSVRowIssue:
    raw: CSVRawRow
    code: str
    message: str
    field_name: str | None = None


@dataclass(frozen=True, slots=True)
class CSVParseResult:
    rows: tuple[CSVParsedRow, ...]
    issues: tuple[CSVRowIssue, ...]
    schema_observation: SourceSchemaObservation


def parse_italian_decimal(value: str) -> Decimal | None:
    """Parse unambiguous Italian decimals such as ``1.234,56``."""

    compact = value.strip()
    if not compact:
        return None
    if _ITALIAN_DECIMAL.fullmatch(compact) is None:
        raise ValueError("expected an Italian decimal such as 1.234,56")
    try:
        return Decimal(compact.replace(".", "").replace(",", "."))
    except InvalidOperation as error:
        raise ValueError("invalid Italian decimal") from error


def parse_aware_datetime(value: str) -> datetime | None:
    """Parse ISO-8601, require an offset, and canonicalise to UTC."""

    compact = value.strip()
    if not compact:
        return None
    if compact.endswith(("Z", "z")):
        compact = compact[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(compact)
    except ValueError as error:
        raise ValueError("expected an ISO-8601 timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamp must include an explicit timezone offset")
    return parsed.astimezone(UTC)


class CSVSourceAdapter:
    """Parse a CSV payload against an explicit contract."""

    def __init__(self, contract: CSVContract) -> None:
        self.contract = contract

    def parse_text(self, payload: str) -> CSVParseResult:
        reader = csv.reader(
            StringIO(payload, newline=""), delimiter=self.contract.delimiter, strict=True
        )
        try:
            headers = tuple(next(reader))
        except StopIteration:
            headers = ()
        except csv.Error as error:
            raise CSVFormatError(f"CSV header cannot be parsed: {error}") from error
        schema_observation = self._validate_headers(headers)

        parsed_rows: list[CSVParsedRow] = []
        issues: list[CSVRowIssue] = []
        try:
            for source_values in reader:
                raw = CSVRawRow(reader.line_num, headers, tuple(source_values))
                if len(source_values) != len(headers):
                    issues.append(
                        CSVRowIssue(
                            raw=raw,
                            code="row_shape",
                            message=f"expected {len(headers)} cells, received {len(source_values)}",
                        )
                    )
                    continue
                try:
                    values = self._normalise_row(raw)
                except CSVFieldError as error:
                    issues.append(
                        CSVRowIssue(
                            raw=raw,
                            code="invalid_field",
                            message=str(error),
                            field_name=error.field_name,
                        )
                    )
                    continue
                parsed_rows.append(CSVParsedRow(raw=raw, values=values))
        except csv.Error as error:
            message = f"CSV body cannot be parsed at line {reader.line_num}: {error}"
            raise CSVFormatError(message) from error
        return CSVParseResult(
            rows=tuple(parsed_rows),
            issues=tuple(issues),
            schema_observation=schema_observation,
        )

    def read_text(self, payload: str, *, is_demo: bool) -> MappedSourceBatch:
        """Map CSV input into the shared ingestion batch, failing closed at file level."""

        try:
            parsed = self.parse_text(payload)
        except CSVHeaderDriftError as error:
            return MappedSourceBatch(
                source=self.contract.source,
                status=SourceReadStatus.FAILED,
                status_detail=str(error)[:2_000],
                schema_observation=error.observation,
            )
        except CSVFormatError as error:
            return MappedSourceBatch(
                source=self.contract.source,
                status=SourceReadStatus.FAILED,
                status_detail=str(error)[:2_000],
            )

        if not parsed.rows and not parsed.issues:
            field_count = len(self.contract.header_mapping)
            return MappedSourceBatch(
                source=self.contract.source,
                status=SourceReadStatus.EMPTY_VERIFIED,
                status_detail=(
                    f"CSV header matched all {field_count} contracted fields; "
                    "the source returned zero data rows"
                ),
                schema_observation=parsed.schema_observation,
            )

        mapped: dict[int, MappedCompanyRecord] = {}
        for issue in parsed.issues:
            quality_issue = QualityIssue(
                severity=IssueSeverity.ERROR,
                code=issue.code,
                message=issue.message[:2_000],
                field=issue.field_name,
                record_index=max(issue.raw.row_number - 2, 0),
            )
            mapped[issue.raw.row_number] = MappedCompanyRecord(
                raw=self._raw_envelope(issue.raw),
                quality=QualityReport(issues=(quality_issue,)),
            )

        for row in parsed.rows:
            raw = self._raw_envelope(row.raw)
            try:
                company = Company.model_validate({**row.values, "is_demo": is_demo})
                mapped[row.raw.row_number] = MappedCompanyRecord(raw=raw, company=company)
            except ValidationError as error:
                first_error = error.errors(include_url=False)[0]
                location = first_error.get("loc", ())
                field_name = str(location[0]) if location else None
                quality_issue = QualityIssue(
                    severity=IssueSeverity.ERROR,
                    code="invalid_company",
                    message=str(error)[:2_000],
                    field=field_name,
                    record_index=max(row.raw.row_number - 2, 0),
                )
                mapped[row.raw.row_number] = MappedCompanyRecord(
                    raw=raw,
                    quality=QualityReport(issues=(quality_issue,)),
                )

        return MappedSourceBatch(
            source=self.contract.source,
            status=SourceReadStatus.SUCCESS,
            records=tuple(mapped[row_number] for row_number in sorted(mapped)),
            schema_observation=parsed.schema_observation,
        )

    def _validate_headers(self, headers: tuple[str, ...]) -> SourceSchemaObservation:
        actual_fields = tuple(sorted(set(headers)))
        duplicated = tuple(sorted(name for name, count in Counter(headers).items() if count > 1))
        missing = tuple(sorted(self.contract.schema_contract.required_fields - set(headers)))
        unexpected = tuple(sorted(set(headers) - self.contract.schema_contract.known_fields))
        issues = [
            QualityIssue(
                severity=IssueSeverity.ERROR,
                code="missing_required_field",
                message=f"Required CSV header is missing: {field_name}",
                field=field_name,
            )
            for field_name in missing
        ]
        issues.extend(
            QualityIssue(
                severity=IssueSeverity.ERROR,
                code="unexpected_field",
                message=f"CSV schema contains an unexpected header: {field_name!r}",
                field=field_name if 0 < len(field_name) <= 128 else None,
            )
            for field_name in unexpected
        )
        issues.extend(
            QualityIssue(
                severity=IssueSeverity.ERROR,
                code="duplicate_header_field",
                message=f"CSV schema contains a duplicate header: {field_name!r}",
                field=field_name if 0 < len(field_name) <= 128 else None,
            )
            for field_name in duplicated
        )
        observation = SourceSchemaObservation(
            source=self.contract.source,
            contract_version=self.contract.schema_contract.version,
            actual_fields=actual_fields,
            quality=QualityReport(issues=tuple(issues)),
        )
        if duplicated or missing or unexpected:
            raise CSVHeaderDriftError(
                missing=missing,
                unexpected=unexpected,
                duplicated=duplicated,
                observation=observation,
            )
        return observation

    def _raw_envelope(self, raw: CSVRawRow) -> RawRecordEnvelope:
        payload: dict[str, object] = dict(raw.as_mapping())
        if len(raw.values) != len(raw.headers):
            payload["__csv_headers__"] = list(raw.headers)
            payload["__csv_values__"] = list(raw.values)

        observed_at: datetime | None = None
        observed_header = next(
            (
                header
                for header, canonical in self.contract.header_mapping.items()
                if canonical == "source_observed_at"
            ),
            None,
        )
        if observed_header is not None:
            observed_value = payload.get(observed_header)
            if isinstance(observed_value, str):
                try:
                    observed_at = parse_aware_datetime(observed_value)
                except ValueError:
                    observed_at = None
        return RawRecordEnvelope(
            source=self.contract.source,
            source_record_id=f"csv-line-{raw.row_number}",
            payload=payload,
            observed_at=observed_at,
        )

    def _normalise_row(self, raw: CSVRawRow) -> Mapping[str, CanonicalValue]:
        source_row = raw.as_mapping()
        values: dict[str, CanonicalValue] = {"source": self.contract.source}
        for source_header, canonical_field in self.contract.header_mapping.items():
            try:
                values[canonical_field] = self._normalise_field(
                    canonical_field, source_row[source_header]
                )
            except (InvalidPIVA, ValueError) as error:
                raise CSVFieldError(canonical_field, str(error)) from error
        for required in sorted(self.contract.required_fields):
            if values.get(required) is None:
                raise CSVFieldError(required, "required CSV field is blank")
        return MappingProxyType(values)

    def _normalise_field(self, field_name: str, value: str) -> CanonicalValue:
        if field_name == "piva":
            return normalize_piva(value)
        if field_name in self.contract.datetime_fields:
            return parse_aware_datetime(value)
        if field_name in self.contract.decimal_fields:
            return parse_italian_decimal(value)
        if field_name in self.contract.integer_fields:
            parsed = parse_italian_decimal(value)
            if parsed is None:
                return None
            if parsed != parsed.to_integral_value():
                raise ValueError("expected an integer")
            return int(parsed)
        stripped = value.strip()
        return stripped or None
