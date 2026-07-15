"""Provider-neutral ingestion contracts and quality gates."""

from list_engine.ingestion.contracts import (
    MappedCompanyRecord,
    MappedSourceBatch,
    RawBatchEnvelope,
    RawRecordEnvelope,
    SchemaContract,
    SourceReadStatus,
    SourceSchemaObservation,
)
from list_engine.ingestion.quality import (
    IssueSeverity,
    QualityIssue,
    QualityReport,
    canonical_payload_hash,
    evaluate_schema,
    validate_json_object,
)

__all__ = [
    "IssueSeverity",
    "MappedCompanyRecord",
    "MappedSourceBatch",
    "QualityIssue",
    "QualityReport",
    "RawBatchEnvelope",
    "RawRecordEnvelope",
    "SchemaContract",
    "SourceReadStatus",
    "SourceSchemaObservation",
    "canonical_payload_hash",
    "evaluate_schema",
    "validate_json_object",
]
