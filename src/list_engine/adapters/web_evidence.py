"""Pure website evidence extraction inspired by PG Omega's stable parser boundary.

This module never fetches a URL. Network/browser acquisition is intentionally separate
so CI can exercise parsing on synthetic HTML and a failed fetch cannot be mistaken for
a verified empty page.
"""

from __future__ import annotations

import json
import re
from html.parser import HTMLParser
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from list_engine.core.piva import InvalidPIVA, normalize_piva

EvidenceKind = Literal["piva", "email", "pec", "phone"]

_EMAIL_RE = re.compile(r"\b[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_PIVA_LABEL_RE = re.compile(
    r"(?:P\.?\s*IVA|PARTITA\s+IVA|VAT(?:\s*ID)?)\s*[:\-]?\s*"
    r"((?:IT\s*)?[0-9][0-9\s.\-_/]{9,20})",
    re.IGNORECASE,
)
_PEC_DOMAINS = (
    ".pec.it",
    ".legalmail.it",
    ".postecert.it",
    ".pec.aruba.it",
)


class WebEvidenceValue(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    kind: EvidenceKind
    raw_value: str = Field(min_length=1)
    normalized_value: str = Field(min_length=1)
    source_url: str = Field(min_length=1)
    extractor: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)


class WebEvidence(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    source_url: str
    values: tuple[WebEvidenceValue, ...]
    json_ld_blocks_seen: int = Field(ge=0)

    def values_for(self, kind: EvidenceKind) -> tuple[WebEvidenceValue, ...]:
        return tuple(value for value in self.values if value.kind == kind)


class _EvidenceHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.text_parts: list[str] = []
        self.mailto_values: list[str] = []
        self.tel_values: list[str] = []
        self.json_ld_blocks: list[str] = []
        self._json_ld_depth = 0
        self._json_ld_buffer: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = {key.lower(): value or "" for key, value in attrs}
        if tag.lower() == "a":
            href = attributes.get("href", "").strip()
            if href.lower().startswith("mailto:"):
                self.mailto_values.append(href[7:].split("?", 1)[0])
            elif href.lower().startswith("tel:"):
                self.tel_values.append(href[4:].split("?", 1)[0])
        if tag.lower() == "script" and attributes.get("type", "").lower() == "application/ld+json":
            self._json_ld_depth = 1
            self._json_ld_buffer = []
        elif self._json_ld_depth:
            self._json_ld_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if not self._json_ld_depth:
            return
        self._json_ld_depth -= 1
        if tag.lower() == "script" or self._json_ld_depth == 0:
            block = "".join(self._json_ld_buffer).strip()
            if block:
                self.json_ld_blocks.append(block)
            self._json_ld_depth = 0
            self._json_ld_buffer = []

    def handle_data(self, data: str) -> None:
        if self._json_ld_depth:
            self._json_ld_buffer.append(data)
        else:
            stripped = data.strip()
            if stripped:
                self.text_parts.append(stripped)


def normalize_italian_phone(raw: str) -> str | None:
    """Return conservative Italian E.164 or ``None`` for an implausible number."""

    digits = re.sub(r"\D", "", raw)
    if digits.startswith("0039"):
        national = digits[4:]
    elif raw.strip().startswith("+39"):
        national = digits[2:]
    else:
        national = digits
    if not 8 <= len(national) <= 11 or not national.startswith(("0", "3")):
        return None
    return f"+39{national}"


def extract_web_evidence(html: str, *, source_url: str) -> WebEvidence:
    """Extract cited, normalised evidence from already-acquired HTML."""

    parser = _EvidenceHTMLParser()
    parser.feed(html)
    visible_text = " ".join(parser.text_parts)
    candidates: list[WebEvidenceValue] = []

    for match in _PIVA_LABEL_RE.finditer(visible_text):
        _append_piva(candidates, match.group(1), source_url, "html:label")
    for raw in parser.mailto_values:
        _append_email(candidates, raw, source_url, "html:mailto", confidence=0.98)
    for raw in _EMAIL_RE.findall(visible_text):
        _append_email(candidates, raw, source_url, "html:text", confidence=0.85)
    for raw in parser.tel_values:
        _append_phone(candidates, raw, source_url, "html:tel", confidence=0.98)

    for block in parser.json_ld_blocks:
        try:
            parsed = json.loads(block)
        except json.JSONDecodeError:
            continue
        for node in _walk_json(parsed):
            for json_key in ("vatID", "taxID"):
                raw_piva = node.get(json_key)
                if isinstance(raw_piva, str):
                    _append_piva(candidates, raw_piva, source_url, f"jsonld:{json_key}")
            raw_email = node.get("email")
            if isinstance(raw_email, str):
                _append_email(candidates, raw_email, source_url, "jsonld:email", 0.95)
            raw_phone = node.get("telephone")
            if isinstance(raw_phone, str):
                _append_phone(candidates, raw_phone, source_url, "jsonld:telephone", 0.95)

    deduplicated: dict[tuple[str, str], WebEvidenceValue] = {}
    for candidate in candidates:
        evidence_key = (candidate.kind, candidate.normalized_value)
        current = deduplicated.get(evidence_key)
        if current is None or candidate.confidence > current.confidence:
            deduplicated[evidence_key] = candidate
    values = tuple(
        sorted(deduplicated.values(), key=lambda value: (value.kind, value.normalized_value))
    )
    return WebEvidence(
        source_url=source_url,
        values=values,
        json_ld_blocks_seen=len(parser.json_ld_blocks),
    )


def _walk_json(value: object, depth: int = 0) -> list[dict[str, object]]:
    if depth > 6:
        return []
    if isinstance(value, list):
        return [node for item in value for node in _walk_json(item, depth + 1)]
    if not isinstance(value, dict):
        return []
    nodes = [value]
    for child in value.values():
        if isinstance(child, (dict, list)):
            nodes.extend(_walk_json(child, depth + 1))
    return nodes


def _append_piva(output: list[WebEvidenceValue], raw: str, source_url: str, extractor: str) -> None:
    try:
        normalized = normalize_piva(raw)
    except InvalidPIVA:
        return
    output.append(
        WebEvidenceValue(
            kind="piva",
            raw_value=raw,
            normalized_value=normalized,
            source_url=source_url,
            extractor=extractor,
            confidence=0.99,
        )
    )


def _append_email(
    output: list[WebEvidenceValue],
    raw: str,
    source_url: str,
    extractor: str,
    confidence: float,
) -> None:
    normalized = raw.removeprefix("mailto:").strip().lower()
    if _EMAIL_RE.fullmatch(normalized) is None:
        return
    domain = normalized.rsplit("@", 1)[1]
    kind: EvidenceKind = "pec" if domain.endswith(_PEC_DOMAINS) else "email"
    output.append(
        WebEvidenceValue(
            kind=kind,
            raw_value=raw,
            normalized_value=normalized,
            source_url=source_url,
            extractor=extractor,
            confidence=confidence,
        )
    )


def _append_phone(
    output: list[WebEvidenceValue],
    raw: str,
    source_url: str,
    extractor: str,
    confidence: float,
) -> None:
    normalized = normalize_italian_phone(raw)
    if normalized is None:
        return
    output.append(
        WebEvidenceValue(
            kind="phone",
            raw_value=raw,
            normalized_value=normalized,
            source_url=source_url,
            extractor=extractor,
            confidence=confidence,
        )
    )
