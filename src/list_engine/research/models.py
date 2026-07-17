"""Trust-boundary models for cited pre-call research dossiers."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from datetime import date
from decimal import Decimal
from types import MappingProxyType
from typing import Literal, Self
from urllib.parse import parse_qsl, urlsplit

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

from list_engine.core.piva import normalize_piva
from list_engine.ingestion.quality import validate_json_object

_EVIDENCE_ID = re.compile(r"[a-z][a-z0-9_-]{1,63}")
_ISSUE_CODE = re.compile(r"[a-z][a-z0-9_]{1,63}")
_SENSITIVE_KEY_TERMS = (
    "accesstoken",
    "apikey",
    "authorization",
    "bearer",
    "clientsecret",
    "cookie",
    "credential",
    "password",
    "privatekey",
    "refreshtoken",
    "secret",
    "sessionkey",
    "signature",
    "token",
)
_MAX_HUB_DATA_BYTES = 50_000
InferenceRationale = Literal[
    "Ipotesi, non fatto: inferenza derivata esclusivamente da un tag esplicito "
    "nell'evidenza verificata."
]
INFERENCE_RATIONALE: InferenceRationale = (
    "Ipotesi, non fatto: inferenza derivata esclusivamente da un tag esplicito "
    "nell'evidenza verificata."
)

SourceKind = Literal[
    "official_registry",
    "company_website",
    "public_directory",
    "job_board",
    "crm",
]

CallCaution = Literal[
    "Non presentare ipotesi come fatti.",
    "Non citare dati assenti dalle fonti verificate.",
    "Non promettere risultati o condizioni finanziarie.",
    "Non formulare giudizi sul merito creditizio.",
]


def _validated_url(value: str) -> str:
    if value != value.strip() or len(value) > 2_048:
        raise ValueError("source URL must not contain outer whitespace or exceed 2,048 chars")
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("source URL must be an absolute HTTP(S) URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("source URLs cannot contain credentials")
    if any(_is_sensitive_key(key) for key, _value in parse_qsl(parsed.query)):
        raise ValueError("source URLs cannot contain credential-like query parameters")
    if parsed.fragment:
        raise ValueError("source URLs cannot contain fragments")
    return value


def _is_sensitive_key(value: str) -> bool:
    snake_case = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value)
    parts = tuple(part for part in re.split(r"[^a-z0-9]+", snake_case.casefold()) if part)
    squashed = "".join(parts)
    return any(term in squashed for term in _SENSITIVE_KEY_TERMS) or any(
        part in {"auth", "key", "sig"} for part in parts
    )


def _freeze_json(value: object) -> object:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _reject_sensitive_keys(value: object, *, path: str = "hub_data") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if _is_sensitive_key(key):
                raise ValueError(f"{path} contains a credential-like key: {key}")
            _reject_sensitive_keys(item, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_sensitive_keys(item, path=f"{path}[{index}]")


class EvidenceItem(BaseModel):
    """A bounded excerpt acquired by deterministic adapter code before the agent runs."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    evidence_id: str
    source_url: str
    source_kind: SourceKind
    title: str = Field(min_length=1, max_length=255)
    excerpt: str = Field(min_length=1, max_length=5_000)
    observed_at: AwareDatetime
    verified: bool
    tags: tuple[str, ...] = Field(default=(), max_length=20)
    valid_until: date | None = None

    @field_validator("evidence_id")
    @classmethod
    def validate_evidence_id(cls, value: str) -> str:
        if _EVIDENCE_ID.fullmatch(value) is None:
            raise ValueError("evidence_id must be a lowercase ASCII identifier")
        return value

    @field_validator("source_url")
    @classmethod
    def validate_source_url(cls, value: str) -> str:
        return _validated_url(value)

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if len(values) != len(set(values)):
            raise ValueError("evidence tags must be unique")
        if any(not value.strip() or value != value.strip() or len(value) > 128 for value in values):
            raise ValueError("evidence tags must be trimmed, non-blank and at most 128 chars")
        structured_prefixes = ("signal:", "persona:", "objection:", "hook:")
        if any(
            tag.startswith(prefix) and not tag.removeprefix(prefix).strip()
            for tag in values
            for prefix in structured_prefixes
        ):
            raise ValueError("structured evidence tags require a non-blank value")
        return values

    @model_validator(mode="after")
    def validate_signal_window(self) -> Self:
        has_signal = any(tag.startswith("signal:") for tag in self.tags)
        if has_signal and self.valid_until is None:
            raise ValueError("signal:* evidence requires valid_until")
        if self.valid_until is not None and self.valid_until < self.observed_at.date():
            raise ValueError("valid_until cannot precede observed_at")
        return self


class ResearchInput(BaseModel):
    """The complete, bounded context visible to a research agent."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    piva: str
    as_of: date
    website: str | None = None
    hub_data: Mapping[str, object] = Field(default_factory=dict)
    evidence: tuple[EvidenceItem, ...] = Field(min_length=1, max_length=50)

    @field_validator("piva", mode="before")
    @classmethod
    def validate_piva(cls, value: object) -> str:
        if not isinstance(value, str):
            raise ValueError("P.IVA must be text")
        return normalize_piva(value)

    @field_validator("website")
    @classmethod
    def validate_website(cls, value: str | None) -> str | None:
        return _validated_url(value) if value is not None else None

    @field_validator("hub_data", mode="before")
    @classmethod
    def validate_hub_data(cls, value: object) -> Mapping[str, object]:
        validated = validate_json_object(value)
        _reject_sensitive_keys(validated)
        serialized = json.dumps(
            validated,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        if len(serialized) > _MAX_HUB_DATA_BYTES:
            raise ValueError(f"hub_data cannot exceed {_MAX_HUB_DATA_BYTES} UTF-8 bytes")
        frozen = _freeze_json(validated)
        if not isinstance(frozen, Mapping):  # pragma: no cover - validated is a dict
            raise AssertionError("validated hub data did not remain a mapping")
        return frozen

    @field_serializer("hub_data")
    def serialize_hub_data(self, value: Mapping[str, object]) -> dict[str, object]:
        thawed = _thaw_json(value)
        if not isinstance(thawed, dict):  # pragma: no cover - field is always a mapping
            raise AssertionError("frozen hub data did not serialize to an object")
        return thawed

    @model_validator(mode="after")
    def validate_unique_evidence(self) -> Self:
        identifiers = [item.evidence_id for item in self.evidence]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("evidence_id values must be unique within a research input")
        if any(item.observed_at.date() > self.as_of for item in self.evidence):
            raise ValueError("research evidence cannot be observed after as_of")
        expired_signals = tuple(
            item.evidence_id
            for item in self.evidence
            if item.verified
            and any(tag.startswith("signal:") for tag in item.tags)
            and item.valid_until is not None
            and item.valid_until < self.as_of
        )
        if expired_signals:
            raise ValueError(
                "verified signal evidence is expired at as_of: " + ", ".join(expired_signals)
            )
        # Pydantic materializes the outer Mapping annotation as a dict after the
        # before-validator. Freeze that final shell too; nested values are already
        # recursively immutable.
        object.__setattr__(self, "hub_data", MappingProxyType(dict(self.hub_data)))
        return self


class CitedClaim(BaseModel):
    """A factual claim that points to an exact excerpt in the input packet."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    claim: str = Field(min_length=1, max_length=500)
    evidence_id: str
    source_url: str
    supporting_excerpt: str = Field(min_length=1, max_length=1_000)
    confidence: float = Field(ge=0, le=1)

    @field_validator("evidence_id")
    @classmethod
    def validate_evidence_id(cls, value: str) -> str:
        if _EVIDENCE_ID.fullmatch(value) is None:
            raise ValueError("evidence_id must be a lowercase ASCII identifier")
        return value

    @field_validator("source_url")
    @classmethod
    def validate_source_url(cls, value: str) -> str:
        return _validated_url(value)


class CitedInference(BaseModel):
    """A labelled hypothesis whose evidence remains inspectable."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    text: str = Field(min_length=1, max_length=500)
    evidence_ids: tuple[str, ...] = Field(min_length=1, max_length=5)
    rationale: InferenceRationale

    @field_validator("evidence_ids")
    @classmethod
    def validate_evidence_ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if len(values) != len(set(values)):
            raise ValueError("inference evidence IDs must be unique")
        if any(_EVIDENCE_ID.fullmatch(value) is None for value in values):
            raise ValueError("inference evidence IDs must be lowercase ASCII identifiers")
        return values


class ResearchDossier(BaseModel):
    """Structured agent output; still untrusted until an eval result passes."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        populate_by_name=True,
    )

    piva: str
    facts: tuple[CitedClaim, ...] = Field(alias="fatti", max_length=3)
    active_signals: tuple[CitedClaim, ...] = Field(alias="segnali_attivi")
    probable_persona: CitedInference | None = Field(alias="persona_probabile")
    opening_hook: str = Field(alias="hook_apertura", min_length=1, max_length=1_000)
    opening_hook_evidence_ids: tuple[str, ...] = Field(min_length=1, max_length=5)
    likely_objection: CitedInference | None = Field(alias="obiezione_probabile")
    what_not_to_say: tuple[CallCaution, ...] = Field(
        alias="cosa_non_dire",
        min_length=1,
        max_length=10,
    )

    @field_validator("piva", mode="before")
    @classmethod
    def validate_piva(cls, value: object) -> str:
        if not isinstance(value, str):
            raise ValueError("P.IVA must be text")
        return normalize_piva(value)

    @field_validator("opening_hook_evidence_ids", "what_not_to_say")
    @classmethod
    def validate_unique_values(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if len(values) != len(set(values)):
            raise ValueError("dossier lists cannot contain duplicates")
        return values


class AgentAttempt(BaseModel):
    """One terminal SDK attempt retained inside the durable generation record."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    attempt: int = Field(ge=1, le=3)
    session_id: str = Field(min_length=1, max_length=255)
    model: str = Field(min_length=1, max_length=128)
    status: Literal["validation_failed", "succeeded"]
    structured_output_json: str = Field(min_length=1, max_length=100_000)
    validation_error: Literal["schema_validation", "piva_mismatch"] | None = None
    cost_usd: Decimal = Field(ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_attempt_state(self) -> Self:
        try:
            output = json.loads(self.structured_output_json)
        except (TypeError, ValueError) as error:
            raise ValueError("attempt structured output must be valid JSON") from error
        if not isinstance(output, dict):
            raise ValueError("attempt structured output must be a JSON object")
        if self.status == "succeeded" and self.validation_error is not None:
            raise ValueError("a successful attempt cannot carry a validation error")
        if self.status == "validation_failed" and self.validation_error is None:
            raise ValueError("a failed validation attempt requires an error code")
        return self


class AgentGeneration(BaseModel):
    """Auditable SDK execution result before eval approval."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    dossier: ResearchDossier
    provider: str = Field(min_length=1, max_length=64)
    requested_model: str = Field(min_length=1, max_length=128)
    model: str = Field(min_length=1, max_length=128)
    prompt_version: str = Field(min_length=1, max_length=64)
    session_id: str = Field(min_length=1, max_length=255)
    cost_usd: Decimal = Field(ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    attempts: tuple[AgentAttempt, ...] = Field(default=(), max_length=3)

    @model_validator(mode="after")
    def validate_attempt_history(self) -> Self:
        if not self.attempts:
            synthetic = AgentAttempt(
                attempt=1,
                session_id=self.session_id,
                model=self.model,
                status="succeeded",
                structured_output_json=self.dossier.model_dump_json(by_alias=True),
                cost_usd=self.cost_usd,
                input_tokens=self.input_tokens,
                output_tokens=self.output_tokens,
            )
            object.__setattr__(self, "attempts", (synthetic,))

        attempts = self.attempts
        if tuple(attempt.attempt for attempt in attempts) != tuple(range(1, len(attempts) + 1)):
            raise ValueError("agent attempt numbers must be contiguous and start at one")
        if any(attempt.status == "succeeded" for attempt in attempts[:-1]):
            raise ValueError("only the final agent attempt can succeed")
        final = attempts[-1]
        if final.status != "succeeded":
            raise ValueError("a generation requires one successful final attempt")
        if final.session_id != self.session_id or final.model != self.model:
            raise ValueError("final attempt metadata must match the generation")
        if sum((attempt.cost_usd for attempt in attempts), start=Decimal(0)) != self.cost_usd:
            raise ValueError("generation cost must equal the sum of attempt costs")

        for field in ("input_tokens", "output_tokens"):
            attempt_values = tuple(getattr(attempt, field) for attempt in attempts)
            expected = (
                None if any(value is None for value in attempt_values) else sum(attempt_values)
            )
            if getattr(self, field) != expected:
                raise ValueError(f"generation {field} must equal complete attempt usage")
        return self


class EvaluationIssue(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    code: str
    message: str = Field(min_length=1, max_length=2_000)
    field: str | None = Field(default=None, min_length=1, max_length=255)

    @field_validator("code")
    @classmethod
    def validate_code(cls, value: str) -> str:
        if _ISSUE_CODE.fullmatch(value) is None:
            raise ValueError("evaluation issue code must be a lowercase ASCII identifier")
        return value


class DossierEvaluation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    passed: bool
    factual_support: float = Field(ge=0, le=1)
    citation_accuracy: float = Field(ge=0, le=1)
    completeness: float = Field(ge=0, le=1)
    source_quality: float = Field(ge=0, le=1)
    hallucination_free: float = Field(ge=0, le=1)
    issues: tuple[EvaluationIssue, ...] = ()
    evaluator_version: str = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def validate_pass_state(self) -> Self:
        if self.passed and self.issues:
            raise ValueError("a passing evaluation cannot contain blocking issues")
        return self


class ApprovedDossier(BaseModel):
    """The only dossier type accepted by downstream delivery code."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    generation: AgentGeneration
    evaluation: DossierEvaluation

    @model_validator(mode="after")
    def require_passing_eval(self) -> Self:
        if not self.evaluation.passed:
            raise ValueError("downstream dossiers require a passing eval gate")
        return self


__all__ = [
    "INFERENCE_RATIONALE",
    "AgentAttempt",
    "AgentGeneration",
    "ApprovedDossier",
    "CallCaution",
    "CitedClaim",
    "CitedInference",
    "DossierEvaluation",
    "EvaluationIssue",
    "EvidenceItem",
    "InferenceRationale",
    "ResearchDossier",
    "ResearchInput",
    "SourceKind",
]
